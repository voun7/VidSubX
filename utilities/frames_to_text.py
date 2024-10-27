import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import paddle
from paddleocr import PaddleOCR

import utilities.utils as utils

OCR_CONFIG = {"use_gpu": utils.Config.use_gpu, "drop_score": utils.Config.text_drop_score, "use_angle_cls": True,
              "lang": utils.Config.ocr_rec_language, "show_log": False}

logger = logging.getLogger(__name__)


def extract_bboxes(files: Path) -> list:
    """
    Returns the bounding boxes of detected texted in images.
    :param files: Directory with images for detection.
    """
    ocr_fn = PaddleOCR(**OCR_CONFIG)
    boxes = []
    for file in files.iterdir():
        result = ocr_fn.ocr(str(file))
        if result := result[0]:
            for line in result:
                box = line[0]
                boxes.append(box)
    return boxes


def extract_text(text_output: Path, files: list) -> None:
    """
    Extract text from a frame using paddle ocr.
    :param text_output: directory for extracted texts.
    :param files: files with text for extraction.
    """
    ocr_fn = PaddleOCR(**OCR_CONFIG)
    for file in files:
        result = ocr_fn.ocr(str(file))
        text = " ".join([line[1][0] for line in result[0]] if result[0] else "")
        with open(f"{text_output}/{file.stem}.txt", 'w', encoding="utf-8") as text_file:
            text_file.write(text)


def frames_to_text(frame_output: Path, text_output: Path) -> None:
    """
    Extracts the texts from frames using multiprocessing
    :param frame_output: directory of the frames
    :param text_output: directory for extracted texts
    """
    chunk_size = utils.Config.text_extraction_chunk_size  # Size of files given to each processor.
    if utils.Config.use_gpu and paddle.device.is_compiled_with_cuda():
        device, max_processes = "GPU", utils.Config.ocr_gpu_max_processes
    else:
        device, max_processes = "CPU", utils.Config.ocr_cpu_max_processes
    prefix = "Text Extraction"
    if utils.Process.interrupt_process:  # Cancel if process has been cancelled by gui.
        logger.warning(f"{prefix} process interrupted!")
        return

    files = list(frame_output.iterdir())
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    no_chunks = len(file_chunks)
    logger.info(f"Starting Multiprocess {prefix} from frames on {device}... "
                f"Processes: {max_processes}, Chunks: {no_chunks}")
    with ProcessPoolExecutor(max_processes) as executor:
        futures = [executor.submit(extract_text, text_output, files) for files in file_chunks]
        for i, f in enumerate(as_completed(futures)):  # as each  process completes
            f.result()  # Prevents silent bugs. Exceptions raised will be displayed.
            utils.print_progress(i, no_chunks - 1, prefix)
    logger.info(f"{prefix} done!")
