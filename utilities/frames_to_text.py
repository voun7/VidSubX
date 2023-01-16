import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import utilities.utils as utils
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

paddle_ocr = PaddleOCR(use_angle_cls=True, lang=utils.Config.ocr_rec_language, drop_score=0.8, show_log=False)


def extract_text(text_output: Path, files: list) -> int:
    """
    Extract text from a frame using paddle ocr
    :param text_output: directory for extracted texts
    :param files: files with text for extraction
    :return: count of texts extracted
    """
    saved_count = 0
    for file in files:
        result = paddle_ocr.ocr(str(file))
        result = result[0]
        if result:
            text_list = [line[1][0] for line in result]
            text = " ".join(text_list)
            name = Path(f"{text_output}/{file.stem}.txt")
            with open(name, 'w', encoding="utf-8") as text_file:
                text_file.write(text)
        saved_count += 1
    return saved_count


def frames_to_text(frame_output: Path, text_output: Path, chunk_size: int = utils.Config.text_extraction_chunk_size,
                   ocr_max_processes: int = utils.Config.ocr_max_processes) -> None:
    """
    Extracts the texts from frames using multiprocessing
    :param frame_output: directory of the frames
    :param text_output: directory for extracted texts
    :param chunk_size: size of files given to each processor
    :param ocr_max_processes: number of processors to be used
    """
    # cancel if process has been cancelled by gui.
    if utils.Process.interrupt_process:
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
