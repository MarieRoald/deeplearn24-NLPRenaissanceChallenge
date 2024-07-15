import logging
from datetime import datetime
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CustomFormatter(logging.Formatter):
    """Log formatter with colorized output.

    Modified from
    CC-BY-SA-4.0: Sergey Pleshakov - https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    """

    grey: str = "\x1b[38;20m"
    blue: str = "\x1b[34;20m"
    yellow: str = "\x1b[33;20m"
    red: str = "\x1b[31;20m"
    bold_red: str = "\x1b[31;1m"
    reset: str = "\x1b[0m"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logging(log_dir: Path = Path("logs"), log_level: int = logging.DEBUG):
    """Setup logging, storing it to a file and printing to the console.

    Modified from
    CC-BY-SA-4.0: Sergey Pleshakov - https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    """
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(CustomFormatter())

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)",
        handlers=[
            logging.FileHandler(log_file),
            ch,
        ],
    )


def download_url(url: str, output_path: Path) -> None:
    """Download a file from a URL and save it to the output path."""
    if output_path.exists():
        logger.info("File already exists: %s, skipping", output_path)
        return
    response = httpx.get(url)
    response.raise_for_status()
    output_path.write_bytes(response.content)
