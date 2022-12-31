import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

paddle_ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)


def extract_text(text_output: Path, files: list) -> int:
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


def frames_to_text(frame_output: Path, text_output: Path, chunk_size: int, ocr_max_processes: int) -> None:
    files = [file for file in frame_output.iterdir()]
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]

    prefix = "Extracting text from frame chunks"
    logger.debug("Using multiprocessing for extracting text")
    with ProcessPoolExecutor(max_workers=ocr_max_processes) as executor:
        futures = [executor.submit(extract_text, text_output, files) for files in file_chunks]
        pbar = tqdm(total=len(file_chunks), desc=prefix, colour="green")
        for f in as_completed(futures):
            error = f.exception()
            if error:
                logger.exception(error)
            pbar.update()
        pbar.close()
    logger.info("Done extracting texts!")
