import logging
import sys
from pathlib import Path
from typing import Generator, Iterable, Literal

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from deeplearn24.types import ImageArray
from doc_ufcn import models
from doc_ufcn.main import DocUFCN
from scipy.ndimage import maximum_filter1d

logging.basicConfig(format="[%(levelname)s] %(message)s", stream=sys.stdout, level=logging.INFO)


def save_polygon_bounding_boxes(detected_polygons: dict, image: ImageArray) -> None:
    """polygon data example:
    {
        1: [
        {
            'confidence': 0.99,
            'polygon': [(490, 140), (490, 1596), (2866, 1598), (2870, 140)]
        }
        ...
        ],
        ...
    }
    """
    bounding_box_vis = image.copy()

    for class_id, polygons in detected_polygons.items():
        for idx, polygon in enumerate(polygons):
            points = polygon["polygon"]
            x, y, w, h = cv2.boundingRect(np.array(points))
            cv2.rectangle(bounding_box_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # crop image using the bounding box
            cropped_image = image[y : y + h, x : x + w]
            # save image as png
            cv2.imwrite(
                str(page_output_directory / f"bounding_box_{class_id}_{idx:03d}.png"),
                cropped_image,
            )
    return bounding_box_vis


def iter_bounding_boxes_from_markers(
    markers: ImageArray, horisontal_padding: int, vertical_padding: int
) -> Generator[tuple[int, tuple[int, int, int, int]], None, None]:
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

        yield marker, (x, y, w, h)


def save_bounding_boxes(
    bounding_boxes: Iterable[tuple[int, tuple[int, int, int, int]]], image: ImageArray
) -> None:
    bounding_box_vis = image.copy()
    for marker, (x, y, w, h) in bounding_boxes:
        if marker <= 0:
            continue

        cv2.rectangle(bounding_box_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # crop image using the bounding box
        cropped_image = image[y : y + h, x : x + w]
        # save image as png
        cv2.imwrite(
            str(page_output_directory / f"bounding_box_marker_{marker:03d}.png"), cropped_image
        )

    return bounding_box_vis


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
    plt.figure()
    plt.plot(-line_sum)
    plt.axhline(ret)
    plt.savefig(str(page_output_directory / f"DEBUG_histogram_{axis}.png"))
    plt.figure()
    plt.plot(-thresh)
    plt.savefig(str(page_output_directory / f"DEBUG_threshold_{axis}.png"))
    plt.close("all")
    cv2.imwrite(str(page_output_directory / f"DEBUG_thresholded_image_{axis}.png"), image)

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
    cv2.imwrite(str(page_output_directory / f"DEBUG_guaranteed_bg.png"), guaranteed_bg)
    cv2.imwrite(str(page_output_directory / f"DEBUG_unknown.png"), unknown)
    cv2.imwrite(str(page_output_directory / f"DEBUG_connected_components.png"), markers)
    cv2.imwrite(str(page_output_directory / f"DEBUG_watershed.png"), markers * 10)


if __name__ == "__main__":
    input_directory = Path("data/1_preprocessing/extracted_pages/processed_page_images")

    output_directory = Path("data/2_bounding_box/Doc-UFCN")
    output_directory.mkdir(exist_ok=True, parents=True)

    image_path = Path(
        "data/1_preprocessing/extracted_pages/processed_page_images/image-002-005_left.png"
    )

    model_path, parameters = models.download_model("generic-historical-line", version="main")

    model = DocUFCN(len(parameters["classes"]), parameters["input_size"], "cpu")
    model.load(model_path, parameters["mean"], parameters["std"])

    for image_path in sorted(input_directory.glob("*.png")):
        print(image_path)
        image_path = Path(image_path)

        page_output_directory = output_directory / image_path.stem
        page_output_directory.mkdir(exist_ok=True, parents=True)

        image = cv2.imread(str(image_path))
        image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image_copy = mask_away_non_roi(image_copy, axis=0, min_size=10)
        image_copy = mask_away_non_roi(image_copy, axis=1)

        detected_polygons, probabilities, mask, overlap = model.predict(
            image_copy, raw_output=True, mask_output=True, overlap_output=True
        )

        # get bounding boxes for the polygons
        # bounding_box_vis = save_polygon_bounding_boxes(detected_polygons, image)

        # Save intermediate states for debugging
        # cv2.imwrite(str(page_output_directory / f"DEBUG_bounding_boxes.png"), bounding_box_vis)
        cv2.imwrite(
            str(page_output_directory / f"DEBUG_countours.png"),
            cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB),
        )
        cv2.imwrite(str(page_output_directory / f"DEBUG_mask.png"), mask)

        # Image analysis post processing of the mask to try to split joined lines
        original_mask = mask.copy()
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 1), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 21), np.uint8), iterations=1)
        cv2.imwrite(str(page_output_directory / f"DEBUG_mask_closed.png"), mask)

        # Erode the mask horisontally to split joined lines
        mask = cv2.erode(mask, np.ones((1, 5), np.uint8), iterations=10)

        # Ensure mask is of type uint8
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        cv2.imwrite(str(page_output_directory / f"DEBUG_mask_processed.png"), mask)

        # Use connected components to label the known blobs
        ret, markers = cv2.connectedComponents(mask)
        bounding_boxes = iter_bounding_boxes_from_markers(
            markers, horisontal_padding=30, vertical_padding=10
        )
        bbox_vis2 = save_bounding_boxes(bounding_boxes, image)
        cv2.imwrite(str(page_output_directory / f"DEBUG_bounding_boxes2.png"), bbox_vis2)
