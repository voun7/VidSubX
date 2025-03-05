import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import onnxruntime as ort
from paddleocr import PaddleOCR

import utilities.utils as utils

logger = logging.getLogger(__name__)


def download_models() -> None:
    """
    Download models if dir does not exist.
    """
    if not utils.Config.model_dir.exists():
        logger.info("Checking for requested models...")
        _ = PaddleOCR(lang=utils.Config.ocr_rec_language, **utils.Config.ocr_opts)
        logger.info("")


def extract_bboxes(files: Path) -> list:
    """
    Returns the bounding boxes of detected texted in images.
    :param files: Directory with images for detection.
    """
    ocr_engine = PaddleOCR(use_gpu=utils.Config.use_gpu, drop_score=utils.Config.text_drop_score,
                           lang=utils.Config.ocr_rec_language, **utils.Config.ocr_opts)
    boxes = []
    for file in files.iterdir():
        result = ocr_engine.ocr(str(file))
        if result := result[0]:
            for line in result:
                box = line[0]
                boxes.append(box)
    return boxes


def extract_text(ocr_engine, text_output: Path, files: list, line_sep: str) -> None:
    """
    Extract text from a frame using ocr.
    :param ocr_engine: OCR Engine.
    :param text_output: directory for extracted texts.
    :param files: files with text for extraction.
    :param line_sep: line seperator for the text.
    """
    for file in files:
        result = ocr_engine.ocr(str(file))
        text = line_sep.join([line[1][0] for line in result[0]] if result[0] else "")
        with open(f"{text_output}/{file.stem}.txt", 'w', encoding="utf-8") as text_file:
            text_file.write(text)


def frames_to_text(frame_output: Path, text_output: Path) -> None:
    """
    Extracts the texts from frames using multiprocessing
    :param frame_output: directory of the frames
    :param text_output: directory for extracted texts
    """
    chunk_size = utils.Config.text_extraction_chunk_size  # Size of files given to each processor.
    if utils.Config.use_gpu and ort.get_device() == "GPU":
        device, max_processes = "GPU", utils.Config.ocr_gpu_max_processes
    else:
        device, max_processes = "CPU", utils.Config.ocr_cpu_max_processes
    prefix = "Text Extraction"
    if utils.Process.interrupt_process:  # Cancel if process has been cancelled by gui.
        logger.warning(f"{prefix} process interrupted!")
        return

    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = max_processes
    ocr_config = {"use_gpu": utils.Config.use_gpu, "drop_score": utils.Config.text_drop_score,
                  "lang": utils.Config.ocr_rec_language, "onnx_sess_options": sess_opt} | utils.Config.ocr_opts
    ocr_engine = PaddleOCR(**ocr_config)
    line_sep = "\n" if utils.Config.line_break else " "
    files = list(frame_output.iterdir())
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    no_chunks = len(file_chunks)
    logger.info(f"Starting Multiprocess {prefix} from frames on {device}... "
                f"Max Processes: {max_processes}, Chunks: {no_chunks}")
    with ThreadPoolExecutor(max_processes) as executor:
        futures = [executor.submit(extract_text, ocr_engine, text_output, files, line_sep) for files in file_chunks]
        for i, f in enumerate(as_completed(futures)):  # as each  process completes
            f.result()  # Prevents silent bugs. Exceptions raised will be displayed.
            utils.print_progress(i, no_chunks - 1, prefix)
    logger.info(f"{prefix} done!")
