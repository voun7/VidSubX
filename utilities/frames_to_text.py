import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from sub_ocr.subtitle_ocr import SubtitleOCR

import utilities.utils as utils

logger = logging.getLogger(__name__)

subtitle_ocr = SubtitleOCR(utils.Config.ocr_rec_language, f"{Path(__file__).parent.parent}/models")


def extract_bboxes(files: Path) -> list:
    """
    Returns the bounding boxes of detected texted in images.
    :param files: Directory with images for detection.
    """
    boxes = []
    for file in files.iterdir():
        result = subtitle_ocr.ocr(str(file))
        for line in result:
            box = line.get("bbox")
            if box and line["score"] > utils.Config.text_drop_score:
                boxes.append(box)
    return boxes


def extract_text(text_output: Path, files: list, drop_score: float) -> None:
    """
    Extract text from a frame using paddle ocr.
    :param text_output: directory for extracted texts.
    :param files: files with text for extraction.
    :param drop_score: Text with a score below the drop score will not be used.
    """
    for file in files:
        result = subtitle_ocr.ocr(str(file))
        text = " ".join([line["text"] for line in result if line["score"] > drop_score])
        name = Path(f"{text_output}/{file.stem}.txt")
        with open(name, 'w', encoding="utf-8") as text_file:
            text_file.write(text)


def frames_to_text(frame_output: Path, text_output: Path) -> None:
    """
    Extracts the texts from frames using multiprocessing
    :param frame_output: directory of the frames
    :param text_output: directory for extracted texts
    """
    chunk_size = utils.Config.text_extraction_chunk_size  # Size of files given to each processor.
    text_drop_score = utils.Config.text_drop_score
    if utils.Config.use_gpu:
        max_processes = utils.Config.ocr_gpu_max_processes
    else:
        max_processes = utils.Config.ocr_cpu_max_processes
    prefix = "Text Extraction"
    if utils.Process.interrupt_process:  # Cancel if process has been cancelled by gui.
        logger.warning(f"{prefix} process interrupted!")
        return

    logger.info(f"Starting {prefix} from frames...")
    files = list(frame_output.iterdir())
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    no_chunks = len(file_chunks)
    logger.debug(f"Using multiprocessing for {prefix}, {max_processes=}, {no_chunks=}")
    with ProcessPoolExecutor(max_processes) as executor:
        futures = [executor.submit(extract_text, text_output, files, text_drop_score) for files in file_chunks]
        for i, f in enumerate(as_completed(futures)):  # as each  process completes
            f.result()  # Prevents silent bugs. Exceptions raised will be displayed.
            utils.print_progress(i, no_chunks - 1, prefix)
    logger.info(f"{prefix} done!")
