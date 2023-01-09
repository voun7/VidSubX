import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import utilities.utils as utils
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

paddle_ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)


def extract_text(text_output: Path, files: list) -> int | None:
    """
    Extract text from a frame using paddle ocr
    :param text_output: directory for extracted texts
    :param files: files with text for extraction
    :return: count of texts extracted
    """
    saved_count = 0
    for file in files:
        saved_count += 1
        name = Path(f"{text_output}/{file.stem}.txt")
        result = paddle_ocr.ocr(str(file), cls=True)
        if result[0]:
            text = result[0][0][1][0]
            with open(name, 'w', encoding="utf-8") as text_file:
                text_file.write(text)
    return saved_count


def frames_to_text(frame_output: Path, text_output: Path, chunk_size: int = 150, ocr_max_processes: int = 4) -> None:
    """
    Extracts the texts from frames using multiprocessing
    :param frame_output: directory of the frames
    :param text_output: directory for extracted texts
    :param chunk_size: size of files given to each processor
    :param ocr_max_processes: number of processors to be used
    """
    # cancel if process has been cancelled.
    if utils.process_state():
        logger.warning("Text extraction process interrupted!")
        return

    logger.info("Starting to extracting text from frames...")

    files = [file for file in frame_output.iterdir()]
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]

    prefix = "Extracting text from frame chunks"
    logger.debug("Using multiprocessing for extracting text")

    with ProcessPoolExecutor(max_workers=ocr_max_processes) as executor:
        futures = [executor.submit(extract_text, text_output, files) for files in file_chunks]
        for i, f in enumerate(as_completed(futures)):  # as each  process completes
            error = f.exception()
            if error:
                logger.exception(f.result())
                logger.exception(error)
            # print it's progress
            utils.print_progress(i, len(file_chunks) - 1, prefix=prefix, suffix='Complete')
    logger.info("Text Extraction Done!")
