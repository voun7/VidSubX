import logging
import shutil
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path

import cv2 as cv
from natsort import natsorted

from frames_to_text import frames_to_text
from video_to_frames import video_to_frames

logger = logging.getLogger(__name__)


class SubtitleExtractor:
    def __init__(self) -> None:
        """
        Extracts hardcoded subtitles from video.
        """
        self.video_path = None
        # Create cache directory
        self.vd_output_dir = Path(f"{Path.cwd()}/output")
        # Extracted video frame storage directory
        self.frame_output = self.vd_output_dir / "frames"
        # Extracted text file storage directory
        self.text_output = self.vd_output_dir / "extracted texts"

    @staticmethod
    def video_details(video_path) -> tuple:
        """
        Get the video details of the video in path.

        :return: video details
        """
        capture = cv.VideoCapture(str(video_path))
        fps = capture.get(cv.CAP_PROP_FPS)
        frame_total = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        return fps, frame_total, frame_width, frame_height

    @staticmethod
    def default_sub_area(frame_width, frame_height, sub_area: None | tuple) -> tuple:
        """
        Returns a default subtitle area that can be used if no subtitle is given.
        :return: Position of subtitle relative to the resolution of the video. x2 = width and y2 = height
        """
        if sub_area:
            return sub_area
        else:
            x1, y1, x2, y2 = 0, int(frame_height * 0.75), frame_width, frame_height
            return x1, y1, x2, y2

    def empty_cache(self) -> None:
        """
        Delete all cache files produced during subtitle extraction.
        """
        if self.vd_output_dir.exists():
            shutil.rmtree(self.vd_output_dir)
            logger.debug("Emptying cache...")

    @staticmethod
    def similarity(text1: str, text2: str, similarity_threshold: float = 0.8) -> float:
        return SequenceMatcher(a=text1, b=text2).quick_ratio() > similarity_threshold

    def remove_duplicate_texts(self) -> None:
        logger.info("Deleting duplicate texts...")
        for file in self.text_output.iterdir():
            if "--" not in file.name:
                file.unlink()

    def merge_similar_texts(self) -> None:
        no_of_files = len(list(self.text_output.iterdir())) - 1
        counter = 0
        starting_file = None
        for file1, file2 in pairwise(natsorted(self.text_output.iterdir())):
            similarity = self.similarity(file1.read_text(encoding="utf-8"), file2.read_text(encoding="utf-8"))
            counter += 1
            if similarity and counter != no_of_files:
                # print(file1.name, file2.name, similarity)
                if not starting_file:
                    starting_file = file1
            else:
                # print(file1.name, file2.name, similarity)
                if not starting_file:
                    starting_file = file1
                ending_file = file1
                if starting_file == ending_file:
                    ending_file = file2
                new_file_name = f"{starting_file.stem}--{ending_file.stem}.txt"
                starting_file.rename(f"{starting_file.parent}/{new_file_name}")
                starting_file = None

    @staticmethod
    def timecode(frame_no: float) -> str:
        seconds = frame_no // 1000
        milliseconds = int(frame_no % 1000)
        minutes = 0
        hours = 0
        if seconds >= 60:
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
        if minutes >= 60:
            hours = int(minutes // 60)
            minutes = int(minutes % 60)
        smpte_token = ','
        return "%02d:%02d:%02d%s%03d" % (hours, minutes, seconds, smpte_token, milliseconds)

    def _save_subtitle(self, lines: list) -> None:
        name = self.video_path.with_suffix(".srt")
        if name.exists():
            name = f"{name.parent}/{name.stem} (new copy).srt"
        logger.info(f"Subtitle file successfully generated. Name: {name}")
        with open(name, 'w', encoding="utf-8") as new_sub:
            new_sub.writelines(lines)

    def generate_subtitle(self) -> None:
        self.merge_similar_texts()
        self.remove_duplicate_texts()
        subtitles = []
        line_code = 0
        for file in natsorted(self.text_output.iterdir()):
            file_name = file.stem.split("--")
            line_code += 1
            frame_start = self.timecode(float(file_name[0]))
            frame_end = self.timecode(float(file_name[1]))
            file_content = file.read_text(encoding="utf-8")
            subtitle_line = f"{line_code}\n{frame_start} --> {frame_end}\n{file_content}\n\n"
            subtitles.append(subtitle_line)
        self._save_subtitle(subtitles)
        logger.info("Subtitle generated!")

    def run(self, video_path: Path, sub_area: tuple = None) -> None:
        """
        Run through the steps of extracting texts from subtitle area in video.
        """
        start = cv.getTickCount()
        # Empty cache at the beginning of program run before it recreates itself
        self.empty_cache()
        # If the directory does not exist, create the folder
        if not self.frame_output.exists():
            self.frame_output.mkdir(parents=True)
        if not self.text_output.exists():
            self.text_output.mkdir(parents=True)

        self.video_path = video_path

        fps, frame_total, frame_width, frame_height = self.video_details(self.video_path)
        sub_area = self.default_sub_area(frame_width, frame_height, sub_area)

        logger.info(f"File Path: {self.video_path}")
        logger.info(f"Frame Total: {frame_total}, Frame Rate: {fps}")
        logger.info(f"Resolution: {frame_width} X {frame_height}")
        logger.info(f"Subtitle Area: {sub_area}")

        logger.info("Starting to extracting video keyframes...")
        video_to_frames(self.video_path, self.frame_output, sub_area)
        logger.info("Starting to extracting text from frames...")
        frames_to_text(self.frame_output, self.text_output)
        logger.info("Generating subtitle...")
        self.generate_subtitle()

        end = cv.getTickCount()
        total_time = (end - start) / cv.getTickFrequency()
        logger.info(f"Subtitle file generated successfully, Total time: {round(total_time, 3)}s\n")
        self.empty_cache()


if __name__ == '__main__':
    logger.debug("Main program Started.")
    test_videos = Path(r"C:\Users\VOUN-XPS\Downloads\test videos")
    se = SubtitleExtractor()
    for video in test_videos.glob("*.mp4"):
        se.run(video)

    logger.debug("Main program Ended.")
