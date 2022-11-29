import logging
import shutil
import time
from pathlib import Path

import cv2
import numpy

from logger_setup import get_log

logger = logging.getLogger(__name__)


class SubtitleExtractor:
    def __init__(self, video_path: Path, sub_area: tuple = None) -> None:
        self.video_path = video_path
        self.video_name = self.video_path.stem
        self.video_cap = cv2.VideoCapture(str(video_path))
        self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.sub_area = self.__subtitle_area(sub_area)
        self.vd_output_dir = Path(f"{Path.cwd()}/output/{self.video_name}")
        # Extracted video frame storage directory
        self.frame_output_dir = self.vd_output_dir / "frames"
        # Extracted subtitle file storage directory
        self.subtitle_output_dir = self.vd_output_dir / "subtitle"
        # If the directory does not exist, create the folder
        if not self.frame_output_dir.exists():
            self.frame_output_dir.mkdir(parents=True)
        if not self.subtitle_output_dir.exists():
            self.subtitle_output_dir.mkdir(parents=True)

    def run(self) -> None:
        """
        Run through the steps of extracting video.
        """
        start = time.perf_counter()
        logger.info(f"File Path: {self.video_path}")
        logger.info(f"Frame Rate: {self.fps}, Frame Count: {self.frame_count}")
        logger.info(f"Resolution: {self.frame_height} X {self.frame_width}")

        logger.info("Start to extracting video keyframes...")
        self.extract_subtitle_frame()

        end = time.perf_counter()
        total_time = end - start
        logger.info(f"Subtitle file generated successfully, Total time: {round(total_time, 3)}s")
        # self.empty_cache()

    def __subtitle_area(self, sub_area: None | tuple) -> tuple:
        """
        Returns a default subtitle area if no subtitle is given.
        :return: A nested tuple containing 2 tuple points
        """
        if not sub_area:
            top, bottom, left, right = 300, 25, 25, 25
            default_sub_area = (left, self.frame_height - top), (self.frame_width - right, self.frame_height - bottom)
            return default_sub_area
        else:
            return sub_area

    @staticmethod
    def rescale_frame(frame: numpy.ndarray, scale=0.5):
        height = int(frame.shape[0] * scale)
        width = int(frame.shape[1] * scale)
        dimensions = (width, height)
        return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

    def extract_subtitle_frame(self) -> None:
        while self.video_cap.isOpened():
            success, frame = self.video_cap.read()
            if not success:
                logger.info(f"Video has ended!")
                break

            color_red = (0, 0, 255)
            cv2.rectangle(frame, self.sub_area[0], self.sub_area[1], color_red, 2)
            frame_resized = self.rescale_frame(frame)
            cv2.imshow("Video Output", frame_resized)

            if cv2.waitKey(4) & 0xFF == ord('d'):
                break

        self.video_cap.release()
        cv2.destroyAllWindows()

    def empty_cache(self) -> None:
        """
        Delete all cache files produced during subtitle extraction.
        """
        if self.vd_output_dir.exists():
            shutil.rmtree(self.vd_output_dir.parent)


def main():
    get_log()
    logger.debug("Logging Started")

    video = Path("tests/anime video-cut.mp4")
    se = SubtitleExtractor(video)
    se.run()

    logger.debug("Logging Ended\n")


if __name__ == '__main__':
    main()
