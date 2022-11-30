import logging
import shutil
from pathlib import Path

import cv2 as cv
import numpy as np

from logger_setup import get_log

logger = logging.getLogger(__name__)


class SubtitleExtractor:
    def __init__(self, video_path: Path, sub_area: tuple = None) -> None:
        self.video_path = video_path
        self.video_name = self.video_path.stem
        self.sub_area = sub_area
        self.video_cap = cv.VideoCapture(str(video_path))
        self.frame_count = self.video_cap.get(cv.CAP_PROP_FRAME_COUNT)
        self.fps = self.video_cap.get(cv.CAP_PROP_FPS)
        self.frame_height = int(self.video_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv.CAP_PROP_FRAME_WIDTH))
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

    def default_subtitle_area(self) -> tuple:
        """
        Returns a default subtitle area that can be used if no subtitle is given.
        :return: Position of subtitle relative to the resolution of the video. x2 = width and y2 = height
        """
        x1, y1, x2, y2 = 0, int(self.frame_height * 0.75), self.frame_width, self.frame_height
        return x1, y1, x2, y2

    @staticmethod
    def rescale_frame(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
        height = int(frame.shape[0] * scale)
        width = int(frame.shape[1] * scale)
        dimensions = (width, height)
        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

    def extract_frames(self) -> None:
        while self.video_cap.isOpened():
            success, frame = self.video_cap.read()
            if not success:
                logger.warning(f"Video has ended!")  # or failed to read
                break
            if self.sub_area:
                x1, y1, x2, y2 = self.sub_area
            else:
                x1, y1, x2, y2 = self.default_subtitle_area()
            # draw rectangle over subtitle area
            top_left_corner = (x1, y1)
            bottom_right_corner = (x2, y2)
            color_red = (0, 0, 255)
            cv.rectangle(frame, top_left_corner, bottom_right_corner, color_red, 2)
            # crop and save subtitle area
            cropped_frame = frame[y1:y2, x1:x2]
            frame_position = self.video_cap.get(cv.CAP_PROP_POS_MSEC)
            frame_name = f"{self.frame_output_dir}/{frame_position}.jpeg"
            cv.imwrite(frame_name, cropped_frame)

            frame_resized = self.rescale_frame(frame)
            cv.imshow("Video Output", frame_resized)

            if cv.waitKey(1) == ord('q'):
                break

        self.video_cap.release()

    def empty_cache(self) -> None:
        """
        Delete all cache files produced during subtitle extraction.
        """
        if self.vd_output_dir.exists():
            shutil.rmtree(self.vd_output_dir.parent)

    def run(self) -> None:
        """
        Run through the steps of extracting video.
        """
        start = cv.getTickCount()
        logger.info(f"File Path: {self.video_path}")
        logger.info(f"Frame Rate: {self.fps}, Frame Count: {self.frame_count}")
        logger.info(f"Resolution: {self.frame_width} X {self.frame_height}")
        logger.info(f"Subtitle Area: {self.sub_area}")

        logger.info("Start to extracting video keyframes...")
        self.extract_frames()

        end = cv.getTickCount()
        total_time = (end - start) / cv.getTickFrequency()
        logger.info(f"Subtitle file generated successfully, Total time: {round(total_time, 3)}s")
        # self.empty_cache()


def main() -> None:
    get_log()
    logger.debug("Logging Started")

    video = Path("tests/I Can Copy Talents.mp4")
    # video = Path("tests/40,000 Years of the Stars.mp4")
    # video = Path("tests/anime video-cut.mp4")
    se = SubtitleExtractor(video)
    se.run()

    logger.debug("Logging Ended\n")


if __name__ == '__main__':
    main()
