import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Event
from pathlib import Path

from tqdm import tqdm

import utilities.utils as utils
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

paddle_ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)


def extract_text(event: Event, text_output: Path, files: list) -> int | None:
    """
    Extract text from a frame using paddle ocr
    :param event: an event used to stop running tasks
    :param text_output: directory for extracted texts
    :param files: files with text for extraction
    :return: count of texts extracted
    """
    saved_count = 0
    for file in files:
        # check if the task should stop
        if event.is_set():
            return

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
    files = [file for file in frame_output.iterdir()]
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]

    prefix = "Extracting text from frame chunks"
    logger.debug("Using multiprocessing for extracting text")

    with Manager() as manager:
        # create an event used to stop running tasks
        event = manager.Event()

        with ProcessPoolExecutor(max_workers=ocr_max_processes) as executor:
            futures = [executor.submit(extract_text, event, text_output, files)
                       for files in file_chunks if not utils.process_state()]
            pbar = tqdm(total=len(file_chunks), desc=prefix, colour="green")
            for f in as_completed(futures):
                error = f.exception()
                if error:
                    logger.exception(error)
                if utils.process_state():
                    logger.warning("Text extraction process interrupted")
                    f.cancel()
                    event.set()
                else:
                    pbar.update()
            pbar.close()
    logger.info("Text Extraction Done!")
