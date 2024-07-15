import logging
from collections import defaultdict
from pathlib import Path

from deeplearn24.utils import download_url, setup_logging
from docx import Document


def is_transcription_paragraph(paragraph: str) -> bool:
    """Check if a paragraph is a transcription paragraph."""
    return paragraph.strip()  # and not paragraph.startswith("PDF p")


if __name__ == "__main__":
    setup_logging(
        Path("./logs/1_preprocessing/extract_page_transcriptions"), log_level=logging.INFO
    )
    transcription_url = "https://raw.githubusercontent.com/ML4SCI/DeepLearnHackathon/main/NLPRenaissanceChallenge/data/Padilla - 1 Nobleza virtuosa_testTranscription.docx"
    transcription_document_path = Path(
        "data/0_input/Padilla - 1 Nobleza virtuosa_testTranscription.docx"
    )
    output_directory = Path("data/1_preprocessing/extracted_transcriptions")

    transcription_document_path.parent.mkdir(exist_ok=True, parents=True)
    download_url(transcription_url, transcription_document_path)

    output_directory.mkdir(exist_ok=True, parents=True)
    document = Document(transcription_document_path)

    paragraphs = defaultdict(list)

    key = "notes"  # the document begins with notes

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        elif text.startswith("PDF p"):
            key = text

        paragraphs[key].append(text)

    for key, paragraph in paragraphs.items():
        file_name = key.lower().replace(" ", "_")
        output_file = output_directory / f"{file_name}.txt"
        output_file.write_text("\n".join(paragraph[1:]))
