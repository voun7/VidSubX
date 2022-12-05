import logging
from pathlib import Path

import numpy as np
from paddleocr import PaddleOCR
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_and_save_text(frame_output: Path, text_output: Path) -> None:
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
    for file in tqdm(frame_output.iterdir(), desc="Extracting texts: "):
        frame = np.load(file)
        name = Path(f"{text_output}/{file.stem}.txt")
        result = ocr.ocr(frame, cls=True)
        if result[0]:
            text = result[0][0][1][0]
            with open(name, 'w', encoding="utf-8") as text_file:
                text_file.write(text)
    logger.info("Done extracting texts!")
