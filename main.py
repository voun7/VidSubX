import logging
import shutil
import time
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path

import cv2 as cv

import utilities.utils as utils
from utilities.frames_to_text import extract_bboxes, frames_to_text
from utilities.logger_setup import get_logger
from utilities.video_to_frames import extract_frames, video_to_frames

logger = logging.getLogger(__name__)


class SubtitleDetector:
    def __init__(self, video_file: str, use_search_area: bool) -> None:
        """
        Detect the subtitle position in a given video using a default sub area as search area.
        :param video_file: The path like string of the video file.
        :param use_search_area: Whether to use the default search area or
        the full video images to search for sub position.
        """
        self.video_file = video_file
        self.use_search_area = use_search_area
        self.sub_ex = SubtitleExtractor()
        self.fps, self.frame_total, self.frame_width, self.frame_height = self.sub_ex.video_details(self.video_file)
        # Create cache directory.
        self.vd_output_dir = Path(f"{Path.cwd()}/output")
        # Extracted video frame storage directory.
        self.frame_output = self.vd_output_dir / "sub detect frames"

    def _get_key_frames(self) -> None:
        """
        Extract frames from default subtitle area of video that should contain subtitles.
        """
        # Decimal used to signify the relative position to choose start point to search for frames.
        split_start = utils.Config.split_start
        # Decimal used to signify the relative position to choose end point to search for frames.
        split_stop = utils.Config.split_stop
        # How many frames to look through after splits.
        no_of_frames = utils.Config.no_of_frames

        relative_start = int(self.frame_total * split_start)
        relative_stop = int(self.frame_total * split_stop)
        logger.debug(f"Relative start frame = {relative_start}, Relative stop frame = {relative_stop}")
        # Split the frames into chunk lists.
        frame_chunks = [[i, i + no_of_frames] for i in range(relative_start, relative_stop)]
        frame_chunks_len = len(frame_chunks)
        logger.debug(f"Frame total = {self.frame_total}, Chunk length = {frame_chunks_len}")
        start_duration = self.sub_ex.frame_no_to_duration(relative_start, self.fps)
        stop_duration = self.sub_ex.frame_no_to_duration(relative_stop, self.fps)
        logger.info(f"Split Start = {start_duration}, Split Stop = {stop_duration}")
        if frame_chunks_len > 3:
            middle_chunk = int(frame_chunks_len / 2)
            frame_chunks = [frame_chunks[0], frame_chunks[middle_chunk], frame_chunks[-1]]
        last_frame_chunk = frame_chunks[-1][-1]
        if last_frame_chunk > self.frame_total:
            frame_chunks[-1][-1] = relative_stop
        logger.debug(f"Frame chunks = {frame_chunks}")
        # Part of the video to look for subtitles.
        if self.use_search_area:
            logger.info("Default sub area is being used as search area.")
            search_area = self.sub_ex.default_sub_area(self.frame_width, self.frame_height)
        else:
            search_area = None
        for frames in frame_chunks:
            extract_frames(self.video_file, self.frame_output, search_area, frames[0], frames[1], int(self.fps))

    def _pad_sub_area(self, top_left: tuple, bottom_right: tuple) -> tuple:
        """
        Prevent boundary box from being too close to text by adding padding.
        """
        x_padding = utils.Config.sub_area_x_rel_padding
        y_padding = utils.Config.sub_area_y_abs_padding
        relative_x_padding = int(self.frame_width * x_padding)
        logger.debug(f"Padding sub area with relative_x_padding = {relative_x_padding}, y_padding = {y_padding}")
        # Use frame width of video to determine width for padding sub area.
        # This makes sure that the width padding will be same for all video resolutions. Relative to width of video.
        top_left = self.frame_width - relative_x_padding, top_left[1] - y_padding
        bottom_right = relative_x_padding, bottom_right[1] + y_padding
        return top_left, bottom_right

    def _reposition_sub_area(self, top_left: tuple, bottom_right: tuple) -> tuple:
        """
        Reposition the sub area that was changed when using the default subtitle area to detect texts bbox.
        """
        if self.use_search_area:
            y = int(self.frame_height * utils.Config.subarea_height_scaler)
            top_left = top_left[0], top_left[1] + y
            bottom_right = bottom_right[0], bottom_right[1] + y
            return top_left, bottom_right
        else:
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
            top_left_x = int(bbox[0][0])
            top_left_y = int(bbox[0][1])
            bottom_right_x = int(bbox[2][0])
            bottom_right_y = int(bbox[2][1])

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
        A more accurate area containing the subtitle in the video is returned.
        """
        video_path = Path(self.video_file)
        new_sub_area = None
        if not video_path.exists():
            logger.error(f"Video file: {video_path.name} ...could not be found!\n")
            return
        # Empty cache at the beginning of program run before it recreates itself.
        self._empty_cache()
        if not self.frame_output.exists():
            self.frame_output.mkdir(parents=True)

        logger.info(f"Video name: {video_path.name}")
        self._get_key_frames()
        bboxes = extract_bboxes(self.frame_output)
        if bboxes:
            top_left, bottom_right = self._get_max_boundaries(bboxes)
            top_left, bottom_right = self._pad_sub_area(top_left, bottom_right)
            top_left, bottom_right = self._reposition_sub_area(top_left, bottom_right)
            new_sub_area = top_left[0], top_left[1], bottom_right[0], bottom_right[1]

        logger.info(f"New sub area = {new_sub_area}\n")
        self._empty_cache()
        return new_sub_area


class SubtitleExtractor:
    def __init__(self) -> None:
        """
        Extracts hardcoded subtitles from video.
        """
        self.video_path = None
        # Create cache directory.
        self.vd_output_dir = Path(f"{Path.cwd()}/output")
        # Extracted video frame storage directory.
        self.frame_output = self.vd_output_dir / "frames"
        # Extracted text file storage directory.
        self.text_output = self.vd_output_dir / "extracted texts"

    @staticmethod
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

    @staticmethod
    def default_sub_area(frame_width: int, frame_height: int) -> tuple:
        """
        Returns a default subtitle area that can be used if no subtitle is given.
        :return: Position of subtitle relative to the resolution of the video. x2 = width and y2 = height
        """
        x1, y1, x2, y2 = 0, int(frame_height * utils.Config.subarea_height_scaler), frame_width, frame_height
        return x1, y1, x2, y2

    def frame_no_to_duration(self, frame_no: float | int, fps: float | int) -> str:
        """
        Covert frame number to milliseconds then to time code duration.
        """
        frame_no_to_ms = (frame_no / fps) * 1000
        duration = self.timecode(frame_no_to_ms).replace(",", ":")
        return duration

    def _empty_cache(self) -> None:
        """
        Delete all cache files produced during subtitle extraction.
        """
        if self.vd_output_dir.exists():
            shutil.rmtree(self.vd_output_dir)
            logger.debug("Emptying cache...")

    def timecode_sort(self, path: Path) -> float:
        """
        Helps sort timecode durations by turning the first value into a float.
        """
        first_timecode = path.stem.split(self.div1)[0]
        return float(first_timecode)

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
        :param divider: Characters for separating time durations in file name.
        """
        logger.debug("Merging adjacent equal texts")
        starting_file, no_of_files = None, len(list(self.text_output.iterdir()))
        for index, (file1, file2) in enumerate(pairwise(sorted(self.text_output.iterdir(), key=self.timecode_sort)),
                                               start=2):
            file1_text, file2_text = file1.read_text(encoding="utf-8"), file2.read_text(encoding="utf-8")
            # print(index, no_of_files, file1.name, file2.name, file1_text, file2_text)
            if file1_text == file2_text and index != no_of_files:
                if not starting_file:
                    starting_file = file1
            else:
                # print("Text not equal\n")
                if not starting_file:  # This condition is used when the file doesn't match the previous or next file.
                    starting_file = file1
                ending_file = file1
                if index == no_of_files:
                    # print("No of files reached!")
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
        Takes a name with two numbers and subtracts to get the duration in milliseconds.
        :param name: Name numbers should seperated by identifier.
        :param divider: Value for splitting string.
        :return: Duration
        """
        name_timecode = name.split(divider)
        duration = float(name_timecode[1]) - float(name_timecode[0])
        return duration

    def _merge_adjacent_similar_texts(self, old_div: str, divider: str) -> None:
        """
        Merge texts that are not the same but beside each other and similar.
        The text that has the longest duration becomes the text for all similar texts.
        :param old_div: Old characters for separating time durations in file name.
        :param divider: Characters for separating time durations in file name.
        """
        logger.debug("Merging adjacent similar texts")
        similarity_threshold = utils.Config.text_similarity_threshold  # Cut off point to determine similarity.
        no_of_files = len(list(self.text_output.iterdir()))
        starting_file = file_text = file_duration = None
        for index, (file1, file2) in enumerate(pairwise(sorted(self.text_output.iterdir(), key=self.timecode_sort)),
                                               start=2):
            file1_text, file1_duration = file1.read_text(encoding="utf-8"), self._name_to_duration(file1.stem, old_div)
            file2_text, file2_duration = file2.read_text(encoding="utf-8"), self._name_to_duration(file2.stem, old_div)
            similarity = self.similarity(file1_text, file2_text)
            # print(f"Index: {index}, No of Files: {no_of_files} "
            #       f"File 1 Name: {file1.name}, Duration: {file1_duration}, Text: {file1_text}\n"
            #       f"File 2 Name: {file2.name}, Duration: {file2_duration}, Text: {file2_text}\n"
            #       f"File 1 & 2 Similarity: {similarity}")
            if similarity >= similarity_threshold and index != no_of_files:
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

                if index == no_of_files:
                    # print("No of files reached!")
                    new_name = file2.name.replace(old_div, divider)
                    new_file_name = f"{self.text_output}/{new_name}"
                    file_text = file2_text
                    # print(f"New file name: {new_file_name} \nNew file text: {file_text}\n")
                    with open(new_file_name, 'w', encoding="utf-8") as text_file:
                        text_file.write(file_text)

                starting_file = file_text = file_duration = None

    @staticmethod
    def delete_files(file_paths: set) -> None:
        for file_path in file_paths:
            Path(file_path).unlink(missing_ok=True)

    def _remove_short_duration_subs(self, divider: str) -> None:
        """
        Deletes files that contain subtitles that have durations that are shorter than the minimum duration
        in consecutive rows.
        :param divider: String in file name that separates the time stamps.
        """
        logger.debug("Removing short duration subs")
        # Minimum allowed time in milliseconds.
        min_sub_duration = utils.Config.min_sub_duration_ms
        # Maximum allowed number of short durations in a row.
        max_consecutive_short_durs = utils.Config.max_consecutive_short_durs

        short_dur_files, no_of_files = set(), len(list(self.text_output.iterdir()))
        for index, (file1, file2) in enumerate(pairwise(sorted(self.text_output.iterdir(), key=self.timecode_sort)),
                                               start=2):
            file1_duration = self._name_to_duration(file1.stem, divider)
            file2_duration = self._name_to_duration(file2.stem, divider)
            # print(f"Index: {index}, No of Files: {no_of_files}, "
            #       f"File 1 Name: {file1.name}, Duration: {file1_duration}\n"
            #       f"File 2 Name: {file2.name}, Duration: {file2_duration}")
            if file1_duration < min_sub_duration and file2_duration < min_sub_duration and index != no_of_files:
                short_dur_files.add(file1)
                short_dur_files.add(file2)
            else:
                if file2_duration < min_sub_duration:
                    short_dur_files.add(file2)
                if len(short_dur_files) >= max_consecutive_short_durs:
                    self.delete_files(short_dur_files)
                    # print(f"Deleting short durations found! Files ({len(short_dur_files)}) = {short_dur_files}\n")
                short_dur_files = set()

    @staticmethod
    def timecode(frame_no_in_milliseconds: float) -> str:
        """
        Use to frame no in milliseconds to create timecode.
        """
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
        """
        # Cancel if process has been cancelled by gui.
        if utils.Process.interrupt_process:
            logger.warning("Subtitle generation process interrupted!")
            return

        logger.info("Generating subtitle...")
        self.div1 = "--"
        div2 = self.div1 + '-'
        self._merge_adjacent_equal_texts(self.div1)
        self._remove_duplicate_texts(self.div1)
        self._merge_adjacent_similar_texts(self.div1, div2)
        self._remove_duplicate_texts(div2)
        self._remove_short_duration_subs(div2)
        subtitles = []
        for line_code, file in enumerate(sorted(self.text_output.iterdir(), key=self.timecode_sort), start=1):
            file_name = file.stem.split(div2)
            frame_start = self.timecode(float(file_name[0]))
            frame_end = self.timecode(float(file_name[1]))
            file_content = file.read_text(encoding="utf-8")
            subtitle_line = f"{line_code}\n{frame_start} --> {frame_end}\n{file_content}\n\n"
            subtitles.append(subtitle_line)
        self._save_subtitle(subtitles)
        logger.info("Subtitle generated!")

    def run_extraction(self, video_path: str, sub_area: tuple = None, start_frame: int = None,
                       stop_frame: int = None) -> None:
        """
        Run through the steps of extracting texts from subtitle area in video to create subtitle.
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            logger.error(f"Video file: {self.video_path.name} ...could not be found!\n")
            return
        start = cv.getTickCount()
        # Empty cache at the beginning of program run before it recreates itself.
        self._empty_cache()
        # If the directory does not exist, create the folder.
        if not self.frame_output.exists():
            self.frame_output.mkdir(parents=True)
        if not self.text_output.exists():
            self.text_output.mkdir(parents=True)

        fps, frame_total, frame_width, frame_height = self.video_details(video_path)

        sub_area = sub_area if sub_area is not None else self.default_sub_area(frame_width, frame_height)

        logger.info(f"File Path: {self.video_path}")
        logger.info(f"Frame Total: {frame_total}, Frame Rate: {fps}")
        logger.info(f"Resolution: {frame_width} X {frame_height}")
        logger.info(f"Subtitle Area: {sub_area}")
        logger.info(f"Start Frame: {start_frame}, Stop Frame: {stop_frame}")

        video_to_frames(video_path, self.frame_output, sub_area, start_frame, stop_frame)
        frames_to_text(self.frame_output, self.text_output)
        self._generate_subtitle()

        end = cv.getTickCount()
        total_time = (end - start) / cv.getTickFrequency()
        converted_time = time.strftime("%Hh:%Mm:%Ss", time.gmtime(total_time))
        logger.info(f"Subtitle Extraction Done! Total time: {converted_time}\n")
        self._empty_cache()


if __name__ == '__main__':
    get_logger()
    logger.debug("\n\nMain program Started.")
    test_video = r""
    test_sub_area = SubtitleDetector(test_video, True).get_sub_area()
    se = SubtitleExtractor()
    se.run_extraction(test_video, test_sub_area)
    logger.debug("Main program Ended.\n\n")
