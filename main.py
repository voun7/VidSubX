import logging
import shutil
from datetime import timedelta
from pathlib import Path

import cv2 as cv
import numpy as np

from frame_ocr import extract_and_save_text
from logger_setup import get_logger
from video_to_frames import video_to_frames

logger = logging.getLogger(__name__)


class SubtitleExtractor:
    def __init__(self, video_path: Path, sub_area: tuple = None) -> None:
        self.video_path = video_path
        self.video_name = self.video_path.stem
        self.video_cap = cv.VideoCapture(str(video_path))
        self.fps = int(self.video_cap.get(cv.CAP_PROP_FPS))
        self.frame_count = int(self.video_cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.frame_height = int(self.video_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.sub_area = self.__subtitle_area(sub_area)
        self.vd_output_dir = Path(f"{Path.cwd()}/output/{self.video_name}")
        # Extracted video frame storage directory
        self.frame_output = self.vd_output_dir / "frames numpy array"
        # Extracted text file storage directory
        self.text_output = self.vd_output_dir / "extracted texts"
        # If the directory does not exist, create the folder
        if not self.frame_output.exists():
            self.frame_output.mkdir(parents=True)
        if not self.text_output.exists():
            self.text_output.mkdir(parents=True)

    def __subtitle_area(self, sub_area: None | tuple) -> tuple:
        """
        Returns a default subtitle area that can be used if no subtitle is given.
        :return: Position of subtitle relative to the resolution of the video. x2 = width and y2 = height
        """
        if sub_area:
            return sub_area
        else:
            x1, y1, x2, y2 = 0, int(self.frame_height * 0.75), self.frame_width, self.frame_height
            return x1, y1, x2, y2

    @staticmethod
    def rescale_frame(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
        height = int(frame.shape[0] * scale)
        width = int(frame.shape[1] * scale)
        dimensions = (width, height)
        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

    def view_frames(self) -> None:
        while self.video_cap.isOpened():
            success, frame = self.video_cap.read()
            if not success:
                logger.warning(f"Video has ended!")  # or failed to read
                break
            x1, y1, x2, y2 = self.sub_area
            # draw rectangle over subtitle area
            top_left_corner = (x1, y1)
            bottom_right_corner = (x2, y2)
            color_red = (0, 0, 255)
            cv.rectangle(frame, top_left_corner, bottom_right_corner, color_red, 2)

            # crop and show subtitle area
            cropped_frame = frame[y1:y2, x1:x2]
            resized_cropped_frame = self.rescale_frame(cropped_frame)
            cv.imshow("Cropped frame", resized_cropped_frame)

            frame_resized = self.rescale_frame(frame)
            cv.imshow("Video Output", frame_resized)

            if cv.waitKey(1) == ord('q'):
                break
        self.video_cap.release()
        cv.destroyAllWindows()

    def generate_subtitle(self):
        position = 1
        for file in sorted(self.text_output.iterdir()):
            file_text = file.read_text(encoding="utf-8")
            name_in_seconds = round(float(file.stem) / 1000, 5)
            print(position, file.name, timedelta(seconds=name_in_seconds), file_text)
            position += 1
        logger.info("Subtitle generated!")

    def empty_cache(self) -> None:
        """
        Delete all cache files produced during subtitle extraction.
        """
        if self.vd_output_dir.exists():
            shutil.rmtree(self.vd_output_dir.parent)
            logger.debug("Emptying cache")

    def run(self) -> None:
        """
        Run through the steps of extracting video.
        """
        start = cv.getTickCount()
        # self.empty_cache()
        logger.info(f"File Path: {self.video_path}")
        logger.info(f"Frame Rate: {self.fps}, Frame Count: {self.frame_count}")
        logger.info(f"Resolution: {self.frame_width} X {self.frame_height}")
        logger.info(f"Subtitle Area: {self.sub_area}")

        # self.view_frames()
        logger.info("Starting to extracting video keyframes...")
        video_to_frames(self.video_path, self.frame_output, self.sub_area, overwrite=False, every=1, chunk_size=250)
        logger.info("Starting to extracting text from frames...")
        extract_and_save_text(self.frame_output, self.text_output)
        logger.info("Generating subtitle...")
        self.generate_subtitle()

        end = cv.getTickCount()
        total_time = (end - start) / cv.getTickFrequency()
        logger.info(f"Subtitle file generated successfully, Total time: {round(total_time, 3)}s")
        self.video_cap.release()
        # self.empty_cache()


if __name__ == '__main__':
    get_logger()
    logger.debug("Logging Started")

    video = Path("tests/I Can Copy Talents.mp4")
    # video = Path("tests/40,000 Years of the Stars.mp4")
    # video = Path("tests/anime video-cut.mp4")
    # video = Path("tests/anime video.mp4")
    se = SubtitleExtractor(video)
    se.run()

    logger.debug("Logging Ended\n")
