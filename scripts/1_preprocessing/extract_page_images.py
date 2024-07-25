import argparse
import logging
import subprocess
from pathlib import Path

import numpy as np
import pymupdf
from deeplearn24.utils import download_url, setup_logging
from deskew import determine_skew
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def pdf_to_images(pdf_path: Path, output_folder: Path) -> None:
    """Converts a PDF file to a series of images.

    Args:
        pdf_path (Path): The path to the PDF file.
        output_folder (Path): The folder where the images will be saved.

    Returns:
        None
    """
    output_folder.mkdir(exist_ok=True, parents=True)
    document = pymupdf.open(pdf_path)

    for page in document:
        pixmap = page.get_pixmap()
        file_name = output_folder / f"{page.number}.png"
        pixmap.save(str(file_name))

    document.close()


def extract_images_from_pdf(pdf_path: Path, output_directory: Path, prefix: str = "image") -> None:
    """
    Extracts images from a PDF file using the `pdfimages` command.

    Args:
        pdf_path (Path): The path to the PDF file.
        output_directory (Path): The directory where the images will be saved.

    Returns:
        None
    """
    output_directory.mkdir(exist_ok=True, parents=True)

    command = ["pdfimages", "-png", "-p", str(pdf_path), f"{output_directory}/{prefix}"]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while extracting images: {e}")


def straighten_image(image: Image.Image) -> Image:
    """
    Straightens an image by rotating it by the angle determined by the `determine_skew` function.

    Args:
        image (Image): The image to be straightened.

    Returns:
        Image: The straightened image.
    """
    angle = determine_skew(np.array(image.convert("L")))
    fillcolor = np.median(np.array(image), axis=(0, 1)).astype(np.uint8)
    return image.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=tuple(fillcolor))


def extract_pages(
    image: Image.Image,
    left_top_left: tuple[int, int],
    left_bottom_right: tuple[int, int],
    right_top_left: tuple[int, int],
    right_bottom_right: tuple[int, int],
    should_straighten: bool,
) -> tuple[Image.Image, Image.Image]:
    """
    Extracts the left and right pages from an image.

    Args:
        image (Image): The image from which the pages will be extracted.
        left_top_left (tuple[int, int]): The coordinates of the top-left corner of the left page.
        left_bottom_right (tuple[int, int]): The coordinates of the bottom-right corner of the left page.
        right_top_left (tuple[int, int]): The coordinates of the top-left corner of the right page.
        right_bottom_right (tuple[int, int]): The coordinates of the bottom-right corner of the right page.
        should_straighten (bool): A flag indicating whether the pages should be straightened.

    Returns:
        tuple[Image, Image]: A tuple containing the left and right pages.
    """
    left_page = image.crop((*left_top_left, *left_bottom_right))
    right_page = image.crop((*right_top_left, *right_bottom_right))

    if should_straighten:
        logger.info("Straightening the pages")
        right_page = straighten_image(right_page)
        left_page = straighten_image(left_page)

    return left_page, right_page


def extract_all_pages(
    file_names: list[Path],
    left_top_left: tuple[int, int],
    left_bottom_right: tuple[int, int],
    right_top_left: tuple[int, int],
    right_bottom_right: tuple[int, int],
    should_straighten: bool,
    output_directory: Path,
) -> None:
    """
    Extracts the left and right pages from a list of images.
    """
    output_directory.mkdir(exist_ok=True, parents=True)
    for file_name in file_names:
        logger.info("Extracting pages from the image: %s", file_name)
        image = Image.open(str(file_name))
        left_page, right_page = extract_pages(
            image,
            left_top_left,
            left_bottom_right,
            right_top_left,
            right_bottom_right,
            should_straighten,
        )

        left_page.save(str(output_directory / f"{file_name.stem}_left.png"))
        right_page.save(str(output_directory / f"{file_name.stem}_right.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pages from a PDF file.")
    parser.add_argument(
        "--pdf_url",
        type=str,
        default="https://raw.githubusercontent.com/ML4SCI/DeepLearnHackathon/main/NLPRenaissanceChallenge/data/Padilla - Nobleza virtuosa_testExtract.pdf",
        help="Path to the PDF file",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="data/1_preprocessing/extracted_pages",
        help="Directory to save the extracted pages",
    )
    parser.add_argument(
        "--disable_straightening",
        action="store_true",
        help="Flag indicating whether to straighten the pages",
    )

    args = parser.parse_args()

    setup_logging(Path("./logs/1_preprocessing/extract_page_images"), log_level=logging.INFO)

    pdf_path = Path("data/0_input/Padilla - Nobleza virtuosa_testExtract.pdf")
    download_url(args.pdf_url, pdf_path)

    output_directory = Path(args.output_directory)
    should_straighten = not args.disable_straightening

    # Area of interest for the left and right pages
    left_top_left = (102, 40)
    left_bottom_right = (620, 828)
    right_top_left = (666, 40)
    right_bottom_right = (1184, 828)

    # Page numbers to extract
    first_page = 1
    last_page = 16
    page_nums = range(first_page, last_page + 1)

    # Directories to save the extracted images
    extracted_raw_images_directory = output_directory / "raw_images"
    extracted_pages_directory = output_directory / "processed_page_images"

    logging.info("Extracting images from the PDF file: %s", pdf_path)
    extract_images_from_pdf(pdf_path, extracted_raw_images_directory)

    file_names = (
        sorted(extracted_raw_images_directory.rglob(f"image-{page_num:03d}-*.png"))[0]
        for page_num in page_nums
    )
    logging.info(
        "Extracting pages from the images in the directory: %s", extracted_raw_images_directory
    )
    extract_all_pages(
        file_names,
        left_top_left,
        left_bottom_right,
        right_top_left,
        right_bottom_right,
        should_straighten,
        extracted_pages_directory,
    )
