import json
import logging
from pathlib import Path
from typing import Literal

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deeplearn24.types import ImageArray
from deeplearn24.utils import setup_logging
from doc_ufcn import models
from doc_ufcn.main import DocUFCN
from scipy.ndimage import maximum_filter1d

logger = logging.getLogger(__name__)

DEBUG = True


def preprocess_image_and_run_doc_ufcn(image: ImageArray) -> ImageArray:
    image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_copy = mask_away_non_roi(image_copy, axis=0, min_size=10)
    image_copy = mask_away_non_roi(image_copy, axis=1)

    # Run Doc-UFCN
    _detected_polygons, _probabilities, mask, _overlap = model.predict(
        image_copy, raw_output=True, mask_output=True, overlap_output=True
    )

    # get bounding boxes for the polygons
    if DEBUG:
        cv2.imwrite(
            str(page_debug_directory / f"DEBUG_countours.png"),
            cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB),
        )
        cv2.imwrite(str(page_debug_directory / f"DEBUG_mask.png"), mask)
    return mask


def get_bounding_boxes_from_markers(
    markers: ImageArray, horisontal_padding: int, vertical_padding: int
) -> dict[int, tuple[int, int, int, int]]:
    out = {}
    for marker in np.unique(markers):
        if marker <= 0:
            continue

        mask = np.zeros_like(markers, dtype=np.uint8)
        mask[markers == marker] = 255
        x, y, w, h = cv2.boundingRect(mask)
        x = max(0, x - horisontal_padding)
        w += horisontal_padding * 2

        y = max(0, y - vertical_padding)
        h += vertical_padding * 2

        out[marker] = (x, y, w, h)

    return out


def draw_bounding_boxes(
    bounding_boxes: dict[int, tuple[int, int, int, int]], image: ImageArray
) -> ImageArray:
    bounding_box_vis = image.copy()
    for marker, (x, y, w, h) in bounding_boxes.items():
        cv2.rectangle(bounding_box_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return bounding_box_vis


def save_bounding_boxes(
    bounding_boxes: dict[int, tuple[int, int, int, int]], image: ImageArray
) -> None:
    bounding_box_vis = image.copy()
    for marker, (x, y, w, h) in bounding_boxes.items():
        # crop image using the bounding box
        cropped_image = image[y : y + h, x : x + w]
        # save image as png
        cv2.imwrite(
            str(page_debug_directory / f"bounding_box_marker_{marker:03d}.png"), cropped_image
        )


def mask_away_non_roi(image: ImageArray, axis: Literal[0, 1], min_size: int = 30) -> ImageArray:
    image = image.copy()
    if axis == 0:
        line_sum = np.sum(image, axis=(1, 2)).astype(float)
    else:  # axis == 1
        line_sum = np.sum(image, axis=(0, 2)).astype(float)

    # Compute the line-wise sum of all pixel values to determine NON-ROI regions
    line_sum = maximum_filter1d(line_sum, min_size).astype(float)
    line_sum *= 255 / np.max(line_sum)
    line_sum = line_sum.astype(np.uint8)

    # Threshold the line sum to get a mask for the NON-ROI regions
    ret, thresh = cv2.threshold(line_sum, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.erode(thresh, np.ones(min_size, np.uint8), iterations=1)
    thresh = thresh.squeeze()
    if axis == 0:
        image[thresh.astype(bool), :, :] = np.median(image, axis=(0, 1))
    else:  # axis == 1
        image[:, thresh.astype(bool), :] = np.median(image, axis=(0, 1))

    # Save debug plots
    if DEBUG:
        plt.figure()
        plt.plot(-line_sum)
        plt.axhline(ret)
        plt.savefig(str(page_debug_directory / f"DEBUG_histogram_{axis}.png"))
        plt.figure()
        plt.plot(-thresh)
        plt.savefig(str(page_debug_directory / f"DEBUG_threshold_{axis}.png"))
        plt.close("all")
        cv2.imwrite(str(page_debug_directory / f"DEBUG_thresholded_image_{axis}.png"), image)

    return image


def run_watershed(mask: ImageArray, processed_mask: ImageArray, image: ImageArray) -> ImageArray:
    """Returns watershed segmentation of the mask (labelled blobs)"""
    mask, processed_mask = mask.copy(), processed_mask.copy()

    # Specify what parts of the image is guaranteed background (for watershed)
    guaranteed_bg = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=10).astype(np.uint8)

    # The current mask is guaranteed to be line-blobs, so to get the undetermined regions,
    # we subtract the mask from the guaranteed background to get the unknown regions
    unknown = cv2.subtract(guaranteed_bg, processed_mask)

    # Add the background as a separate blob and set the unknown regions as background before watershed
    markers += 1
    markers[unknown == 255] = 0

    # Run watershed to grow the mask
    markers = cv2.watershed(image, markers)

    # Save debug images
    if DEBUG:
        cv2.imwrite(str(page_debug_directory / f"DEBUG_guaranteed_bg.png"), guaranteed_bg)
        cv2.imwrite(str(page_debug_directory / f"DEBUG_unknown.png"), unknown)
        cv2.imwrite(str(page_debug_directory / f"DEBUG_connected_components.png"), markers)
        cv2.imwrite(str(page_debug_directory / f"DEBUG_watershed.png"), markers * 10)


def get_page_number(image_path: Path) -> int:
    is_right = "right" in image_path.stem
    image_number = int(image_path.stem.split("-")[1])

    if is_right:
        page_number = (image_number - 1) * 2 + 1
    else:
        page_number = (image_number - 1) * 2
    return page_number


def cleanup_mask(
    mask: ImageArray, closed_mask_filename: Path, processed_mask_filename: Path
) -> ImageArray:
    # Create padded image to prevent boundary issues with the morphological operators
    width, height = mask.shape
    padded_mask = np.zeros((width * 3, height * 3), dtype=mask.dtype)
    padded_mask[width : 2 * width, height : 2 * height] = mask

    # Denoise using an opening operator
    padded_mask = cv2.morphologyEx(
        padded_mask, cv2.MORPH_OPEN, np.ones((5, 1), np.uint8), borderValue=0, iterations=1
    )

    # Close vertical gaps
    padded_mask = cv2.morphologyEx(
        padded_mask,
        cv2.MORPH_CLOSE,
        np.ones((1, 201), np.uint8),
        borderValue=0,
        iterations=1,
    )

    # Remove noise
    padded_mask = cv2.morphologyEx(
        padded_mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        borderValue=0,
        iterations=1,
    )

    if DEBUG:
        cv2.imwrite(str(closed_mask_filename), padded_mask)

    # Erode the mask horisontally to split joined lines
    padded_mask = cv2.erode(padded_mask, np.ones((1, 5), np.uint8), iterations=16)

    # "Unpad the mask"
    mask = padded_mask[width : 2 * width, height : 2 * height]

    # Ensure mask is of type uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    if DEBUG:
        cv2.imwrite(str(processed_mask_filename), mask)

    return mask


def sort_boxes_by_vertical_position(
    bounding_boxes: dict[int, tuple[float, float, float, float]],
) -> dict[int, tuple[float, float, float, float]]:
    sorted_markers = sorted(bounding_boxes, key=lambda m: bounding_boxes[m][1])
    return {m: bounding_boxes[m] for m in sorted_markers}


def save_bounding_boxes_with_transcription(
    bounding_boxes: dict[int, tuple[float, float, float, float]],
    transcription_lines: list[str] | None,
    output_directory: Path,
    line_count: int,
    spread_number: int,
    spread_side: Literal["left", "right"],
    split: Literal["train", "test", "val"],
) -> tuple[pd.DataFrame, int]:
    rows = []

    for _marker, (x, y, w, h) in bounding_boxes.items():
        if transcription_lines is None:
            transcription = None
        else:
            transcription = transcription_lines[line_count - 1]

        logger.debug(f"Line {line_count}: {x=}, {y=}, {w=}, {h=}")
        logger.debug(f"Trancsription: {transcription}")

        # If area of box is too small, ignore it
        logger.debug("Bounding box width: %d, height: %d, area: %d", w, h, w * h)
        if w * h < 1000 and x > 100:
            continue

        # If the box is to high up and likely to be a header, ignore it
        if y < 50:
            continue

        # If the box is to far down to and likely to be a footer, ignore it
        if y > 770:
            continue

        # crop image using the bounding box
        cropped_image = image[y : y + h, x : x + w]

        # save image as png
        bbox_filename = f"image_{spread_number}_{spread_side}_{line_count:03d}.png"
        bbox_path = output_directory / split / bbox_filename
        bbox_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(bbox_path), cropped_image)

        row = {
            "file_name": str(bbox_path.relative_to(output_directory)),
            "spread_number": spread_number,
            "spread_side": spread_side,
            "split": split,
            "line_number": line_count,
            "transcription": transcription,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        }
        rows.append(row)

        line_count += 1

    logger.debug("Transcription info\n%s", rows)
    return pd.DataFrame(rows).set_index("file_name"), line_count


if __name__ == "__main__":
    setup_logging(log_dir=Path("logs/2_bounding_box/Doc-UFCN"), log_level=logging.DEBUG)

    input_directory = Path("data/1_preprocessing/extracted_pages/processed_page_images")
    output_directory = Path("data/2_bounding_box/Doc-UFCN_processed")
    output_directory.mkdir(exist_ok=True, parents=True)

    with Path("data/0_input/data_splits.json").open("r") as f:
        split = json.load(f)
    split_type_map = {
        spread_number: split_type
        for split_type, spread_numbers in split.items()
        for spread_number in spread_numbers
    }

    model_path, parameters = models.download_model("generic-historical-line", version="main")

    model = DocUFCN(len(parameters["classes"]), parameters["input_size"], "cpu")
    model.load(model_path, parameters["mean"], parameters["std"])

    image_paths = sorted(input_directory.glob("*.png"), key=get_page_number)

    page_data = []
    for spread_number in range(1, 17):
        if spread_number in split["test"]:
            transcription_lines = None
        else:
            file_name = f"pdf_p{spread_number:d}.txt"
            transcription_path = Path("data/1_preprocessing/extracted_transcriptions") / file_name
            transcription_lines = transcription_path.read_text().splitlines()

        logger.debug("Transcription lines:\n%s", transcription_lines)

        spread = ["left", "right"] if spread_number > 1 else ["right"]
        bounding_box_list = []
        line_count = 1

        for page in spread:
            image_path = (
                input_directory / f"image-{spread_number:03d}-{(spread_number-1)*5:03d}_{page}.png"
            )
            logger.info("Processing %s", image_path)

            page_debug_directory = (
                output_directory.parent / f"{output_directory.name}_debug" / image_path.stem
            )
            page_debug_directory.mkdir(exist_ok=True, parents=True)

            # Preprocess image
            image = cv2.imread(str(image_path))
            mask = preprocess_image_and_run_doc_ufcn(image)
            mask = cleanup_mask(
                mask,
                closed_mask_filename=page_debug_directory / f"DEBUG_mask_closed.png",
                processed_mask_filename=page_debug_directory / f"DEBUG_mask_processed.png",
            )

            # Use connected components to label the known blobs
            ret, markers = cv2.connectedComponents(mask)
            bounding_boxes = get_bounding_boxes_from_markers(
                markers, horisontal_padding=30, vertical_padding=10
            )
            bounding_boxes = sort_boxes_by_vertical_position(bounding_boxes)

            if DEBUG:
                bbox_vis = draw_bounding_boxes(bounding_boxes, image)
                cv2.imwrite(str(page_debug_directory / f"DEBUG_bounding_boxes2.png"), bbox_vis)

            df, line_count = save_bounding_boxes_with_transcription(
                bounding_boxes=bounding_boxes,
                transcription_lines=transcription_lines,
                output_directory=output_directory,
                line_count=line_count,
                spread_number=spread_number,
                spread_side=page,
                split=split_type_map[spread_number],
            )
            page_data.append(df)

    dataset = pd.concat(page_data)
    dataset.to_csv(output_directory / f"metadata.csv")
