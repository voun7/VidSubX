import logging
import shutil
import time
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path

import cv2 as cv
from natsort import natsorted

import utilities.utils as utils
from utilities.frames_to_text import extract_bboxes, frames_to_text
from utilities.logger_setup import get_logger
from utilities.video_to_frames import extract_frames, video_to_frames

logger = logging.getLogger(__name__)


def video_details(video_path: str) -> tuple:
    """
    Get the video details of the video in path.
    :return: video details
    """
    capture = cv.VideoCapture(video_path)
    fps = capture.get(cv.CAP_PROP_FPS)
    frame_total = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return fps, frame_total, frame_width, frame_height


def default_sub_area(frame_width, frame_height, sub_area: None | tuple) -> tuple:
    """
    Returns a default subtitle area that can be used if no subtitle is given.
    :return: Position of subtitle relative to the resolution of the video. x2 = width and y2 = height
    """
    if sub_area:
        return sub_area
    else:
        logger.debug("Subtitle area being set to default sub area")
        x1, y1, x2, y2 = 0, int(frame_height * 0.75), frame_width, frame_height
        return x1, y1, x2, y2


class SubtitleDetector:
    def __init__(self, video_file: str) -> None:
        self.video_file = video_file
        self.fps, self.frame_total, self.frame_width, self.frame_height = video_details(self.video_file)
        # Create cache directory
        self.vd_output_dir = Path(f"{Path.cwd()}/output")
        # Extracted video frame storage directory
        self.frame_output = self.vd_output_dir / "sub detect frames"

    def get_key_frames(self) -> None:
        """
        Extract specific parts of video that may contain subtitles.
        """
        # value used to divide total frame to choose start point.
        split_start = utils.Config.split_start
        # value used to divide total frame to choose end point.
        split_stop = utils.Config.split_stop
        # how many frame to look through after splits.
        no_of_frames = utils.Config.no_of_frames

        start = int(self.frame_total / split_start)
        stop = int(self.frame_total / split_stop)
        step = no_of_frames * split_start
        # split the frames into chunk lists.
        frame_chunks = [[i, i + no_of_frames] for i in range(start, stop, step)]
        frame_chunks_len = len(frame_chunks)
        if frame_chunks_len > 3:
            middle_chunk = int(frame_chunks_len / 2)
            frame_chunks = [frame_chunks[0], frame_chunks[middle_chunk], frame_chunks[-1]]
        logger.debug(f"Frame total = {self.frame_total}, start = {start}, stop = {stop}, step = {step}")
        logger.debug(f"Frame chunks = {frame_chunks}")
        # part of the video to look for texts.
        key_area = default_sub_area(self.frame_width, self.frame_height, None)

        for frames in frame_chunks:
            extract_frames(self.video_file, self.frame_output, key_area, frames[0], frames[1], self.fps)

    def _pad_sub_area(self, top_left: tuple, bottom_right: tuple) -> tuple:
        """
        Prevent boundary box from being too close to text by adding padding.
        """
        x_padding = utils.Config.sub_area_x_padding
        y_padding = utils.Config.sub_area_y_padding
        logger.debug(f"Padding sub area: top_left = {top_left} and bottom_right = {bottom_right} "
                     f"with x_padding = {x_padding}, y_padding = {y_padding}")
        top_left = x_padding, top_left[1] - y_padding
        bottom_right = self.frame_width - x_padding, bottom_right[1] + y_padding
        return top_left, bottom_right

    def _reposition_sub_area(self, top_left: tuple, bottom_right: tuple) -> tuple:
        """
        Reposition the sub area that was change when using key area to detect texts bbox.
        """
        y = int(self.frame_height * 0.75)
        top_left = top_left[0], top_left[1] + y
        bottom_right = bottom_right[0], bottom_right[1] + y
        return top_left, bottom_right

    def _empty_cache(self) -> None:
        """
        Delete all cache files produced during subtitle extraction.
        """
        if self.vd_output_dir.exists():
            shutil.rmtree(self.vd_output_dir)
            logger.debug("Emptying cache...")

    @staticmethod
    def _get_max_boundaries(bboxes: list) -> tuple:
        """
        Look through all the boundary boxes and use the max value to increase the new boundary size.
        """
        new_top_left_x = new_top_left_y = new_bottom_right_x = new_bottom_right_y = None
        for bbox in bboxes:
            top_left_x = int(bbox[0][0][0])
            top_left_y = int(bbox[0][0][1])
            bottom_right_x = int(bbox[0][2][0])
            bottom_right_y = int(bbox[0][2][1])

            if not new_top_left_x or top_left_x < new_top_left_x:
                new_top_left_x = top_left_x
            if not new_top_left_y or top_left_y < new_top_left_y:
                new_top_left_y = top_left_y
            if not new_bottom_right_x or bottom_right_x > new_bottom_right_x:
                new_bottom_right_x = bottom_right_x
            if not new_bottom_right_y or bottom_right_y > new_bottom_right_y:
                new_bottom_right_y = bottom_right_y
        return (new_top_left_x, new_top_left_y), (new_bottom_right_x, new_bottom_right_y)

    def get_sub_area(self) -> tuple | None:
        """
        Returns the area containing the subtitle in the video.
        """
        # Empty cache at the beginning of program run before it recreates itself.
        self._empty_cache()
        if not self.frame_output.exists():
            self.frame_output.mkdir(parents=True)

        self.get_key_frames()
        bboxes = extract_bboxes(self.frame_output)
        logger.debug(f"bboxes = {bboxes}")

        if bboxes:
            top_left, bottom_right = self._get_max_boundaries(bboxes)
            top_left, bottom_right = self._pad_sub_area(top_left, bottom_right)
            top_left, bottom_right = self._reposition_sub_area(top_left, bottom_right)
            self._empty_cache()
            return top_left[0], top_left[1], bottom_right[0], bottom_right[1]
        else:
            self._empty_cache()
            return None


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

    def _empty_cache(self) -> None:
        """
        Delete all cache files produced during subtitle extraction.
        """
        if self.vd_output_dir.exists():
            shutil.rmtree(self.vd_output_dir)
            logger.debug("Emptying cache...")

    def _remove_duplicate_texts(self, divider: str) -> None:
        """
        Remove all texts from text output that don't have the given divider in their name.
        """
        logger.debug("Deleting duplicate texts...")
        for file in self.text_output.iterdir():
            if divider not in file.name:
                file.unlink()

    def _merge_adjacent_equal_texts(self, divider: str) -> None:
        """
        Merge texts that are beside each other and are the exact same.
        Use divider for duration in text name.
        :param divider: characters for separating time durations in file name
        """
        logger.debug("Merging adjacent equal texts")
        no_of_files = len(list(self.text_output.iterdir()))
        counter = 1
        starting_file = None
        for file1, file2 in pairwise(natsorted(self.text_output.iterdir())):
            file1_text = file1.read_text(encoding="utf-8")
            file2_text = file2.read_text(encoding="utf-8")
            counter += 1
            # print(file1.name, file2.name, file1_text, file2_text)
            if file1_text == file2_text and counter != no_of_files:
                if not starting_file:
                    starting_file = file1
            else:
                # print("Text not equal\n")
                if not starting_file:  # This condition is used when the file doesn't match the previous or next file.
                    starting_file = file1
                ending_file = file1
                if counter == no_of_files:
                    ending_file = file2
                new_file_name = f"{starting_file.stem}{divider}{ending_file.stem}.txt"
                starting_file.rename(f"{starting_file.parent}/{new_file_name}")
                starting_file = None

    @staticmethod
    def similarity(text1: str, text2: str) -> float:
        return SequenceMatcher(a=text1, b=text2).quick_ratio()

    @staticmethod
    def _similar_text_name_gen(start_name: str, end_name: str, divider: str, old_divider) -> str:
        """
        Takes 2 file name durations and creates a new file name.
        """
        start_name = start_name.split(old_divider)[0]
        end_name = end_name.split(old_divider)[1]
        new_name = f"{start_name}{divider}{end_name}.txt"
        return new_name

    @staticmethod
    def _name_to_duration(name: str, divider: str) -> float:
        """
        Takes a name with two numbers and subtracts to get the duration.
        :param name: name numbers should seperated by identifier.
        :param divider: value for splitting string.
        :return: duration
        """
        name_timecode = name.split(divider)
        duration = float(name_timecode[1]) - float(name_timecode[0])
        return duration

    def _merge_adjacent_similar_texts(self, old_div: str, divider: str) -> None:
        """
        Merge texts that are not the same but beside each other and similar.
        The text that has the longest duration becomes the text for all similar texts.
        :param old_div: old characters for separating time durations in file name
        :param divider: characters for separating time durations in file name
        """
        logger.debug("Merging adjacent similar texts")
        # cut off point to determine similarity.
        similarity_threshold = utils.Config.text_similarity_threshold
        no_of_files = len(list(self.text_output.iterdir()))
        counter = 1
        starting_file = file_text = file_duration = None
        for file1, file2 in pairwise(natsorted(self.text_output.iterdir())):
            file1_text, file1_duration = file1.read_text(encoding="utf-8"), self._name_to_duration(file1.stem, old_div)
            file2_text, file2_duration = file2.read_text(encoding="utf-8"), self._name_to_duration(file2.stem, old_div)
            similarity = self.similarity(file1_text, file2_text)
            counter += 1
            # print(f"File 1 Name: {file1.name}, Duration: {file1_duration}, Text: {file1_text}\n"
            #       f"File 2 Name: {file2.name}, Duration: {file2_duration}, Text: {file2_text}\n"
            #       f"File 1 & 2 Similarity: {similarity}")
            if similarity >= similarity_threshold and counter != no_of_files:
                if not starting_file:
                    starting_file = file1
                    file_text = file1_text
                    file_duration = file1_duration

                if file2_duration > file_duration:  # Change text and duration when longer duration is found.
                    # print(f"--- Longer duration found: {file2_duration} ---")
                    file_text = file2_text
                    file_duration = file2_duration
            else:
                if not starting_file:  # This condition is used when the file doesn't match the previous or next file.
                    starting_file = file1
                    file_text = file1_text

                ending_file = file1

                new_name = self._similar_text_name_gen(starting_file.stem, ending_file.stem, divider, old_div)
                new_file_name = f"{self.text_output}/{new_name}"
                # print(f"New file name: {new_file_name} \nNew file text: {file_text}\n")
                with open(new_file_name, 'w', encoding="utf-8") as text_file:
                    text_file.write(file_text)

                if counter == no_of_files:
                    new_name = file2.name.replace(old_div, divider)
                    new_file_name = f"{self.text_output}/{new_name}"
                    file_text = file2_text
                    # print(f"New file name: {new_file_name} \nNew file text: {file_text}\n")
                    with open(new_file_name, 'w', encoding="utf-8") as text_file:
                        text_file.write(file_text)

                starting_file = file_text = file_duration = None

    def _remove_short_duration_subs(self, divider: str, minimum_duration: int = 150) -> None:
        """
        Deletes subtitles that have durations that are shorter than the minimum duration.
        :param divider: string in file name that separates the time stamps.
        :param minimum_duration: Minimum allowed time in milliseconds.
        """
        for file in self.text_output.iterdir():
            duration = self._name_to_duration(file.stem, divider)
            if duration <= minimum_duration:
                # print(f"Deleting short duration found. \nFile name: {file.name}, \nDuration: {duration}\n")
                file.unlink()

    @staticmethod
    def timecode(frame_no_in_milliseconds: float) -> str:
        seconds = frame_no_in_milliseconds // 1000
        milliseconds = int(frame_no_in_milliseconds % 1000)
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
            current_time = time.strftime("%H;%M;%S")
            name = f"{name.parent}/{name.stem} {current_time} (new copy).srt"
        with open(name, 'w', encoding="utf-8") as new_sub:
            new_sub.writelines(lines)
        logger.info(f"Subtitle file generated. Name: {name}")

    def _generate_subtitle(self) -> None:
        """
        Use text files in folder to create subtitle file.
        :return:
        """
        # cancel if process has been cancelled by gui.
        if utils.Process.interrupt_process:
            logger.warning("Subtitle generation process interrupted!")
            return

        logger.info("Generating subtitle...")
        div1 = "--"
        div2 = "---"
        self._merge_adjacent_equal_texts(div1)
        self._remove_duplicate_texts(div1)
        self._merge_adjacent_similar_texts(div1, div2)
        self._remove_duplicate_texts(div2)
        self._remove_short_duration_subs(div2)
        subtitles = []
        line_code = 0
        for file in natsorted(self.text_output.iterdir()):
            file_name = file.stem.split(div2)
            line_code += 1
            frame_start = self.timecode(float(file_name[0]))
            frame_end = self.timecode(float(file_name[1]))
            file_content = file.read_text(encoding="utf-8")
            subtitle_line = f"{line_code}\n{frame_start} --> {frame_end}\n{file_content}\n\n"
            subtitles.append(subtitle_line)
        self._save_subtitle(subtitles)
        logger.info("Subtitle generated!")

    def run(self, video_path: str, sub_area: tuple = None) -> None:
        """
        Run through the steps of extracting texts from subtitle area in video to create subtitle.
        """
        start = cv.getTickCount()
        # Empty cache at the beginning of program run before it recreates itself.
        self._empty_cache()
        # If the directory does not exist, create the folder.
        if not self.frame_output.exists():
            self.frame_output.mkdir(parents=True)
        if not self.text_output.exists():
            self.text_output.mkdir(parents=True)

        self.video_path = Path(video_path)

        fps, frame_total, frame_width, frame_height = video_details(video_path)
        sub_area = default_sub_area(frame_width, frame_height, sub_area)

        logger.info(f"File Path: {self.video_path}")
        logger.info(f"Frame Total: {frame_total}, Frame Rate: {fps}")
        logger.info(f"Resolution: {frame_width} X {frame_height}")
        logger.info(f"Subtitle Area: {sub_area}")

        video_to_frames(self.video_path, self.frame_output, sub_area)
        frames_to_text(self.frame_output, self.text_output)
        self._generate_subtitle()

        end = cv.getTickCount()
        total_time = (end - start) / cv.getTickFrequency()
        logger.info(f"Subtitle Extraction Done! Total time: {round(total_time, 3)}s\n")
        self._empty_cache()


if __name__ == '__main__':
    get_logger()
    logger.debug("\n\nMain program Started.")
    test_video = Path(r"")
    se = SubtitleExtractor()
    se.run(str(test_video))
    logger.debug("Main program Ended.\n\n")
