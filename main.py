import logging
import shutil
import time
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path

import cv2 as cv

import utilities.utils as utils
from utilities.frames_to_text import extract_bboxes, frames_to_text
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
        self.vd_output_dir = Path(f"{Path.cwd()}/output")  # Create cache directory.
        self.frame_output = self.vd_output_dir / "sub detect frames"  # Extracted video frame storage directory.

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

        relative_start, relative_stop = int(self.frame_total * split_start), int(self.frame_total * split_stop)
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
        The x paddings are relative to the width of the video. Different resolutions will have different x paddings.
        The y paddings are absolute to the height of the vide. All resolutions will have the same y paddings.
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
            logger.debug("Emptying cache...")
            shutil.rmtree(self.vd_output_dir)

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
        if not video_path.exists() or not video_path.is_file():
            logger.error(f"Video file: {video_path.name} ...could not be found!\n")
            return
        self._empty_cache()  # Empty cache at the beginning of program run before it recreates itself.
        if not self.frame_output.exists():
            self.frame_output.mkdir(parents=True)

        logger.info(f"Video name: {video_path.name}")
        self._get_key_frames()
        bboxes = extract_bboxes(self.frame_output)
        new_sub_area = None
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
        self.subtitle_texts = {}
        self.divider = "--"  # Characters for separating time durations(ms) in key name.
        self.vd_output_dir = Path(f"{Path.cwd()}/output")  # Create cache directory.
        # Extracted video frame storage directory. Extracted text file storage directory.
        self.frame_output, self.text_output = self.vd_output_dir / "frames", self.vd_output_dir / "extracted texts"

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
        Delete all cache files and dictionary content produced during subtitle extraction.
        """
        if self.vd_output_dir.exists():
            logger.debug("Emptying cache...")
            shutil.rmtree(self.vd_output_dir)
            self.subtitle_texts = {}

    def merge_adjacent_equal_texts(self) -> None:
        """
        Merge texts that are beside each other and are the exact same.
        Use divider for duration in text name.
        """
        logger.debug("Merging adjacent equal texts")
        new_subtitle_dict, starting_key, no_of_keys = {}, None, len(self.subtitle_texts)
        for index, (key1, key2) in enumerate(pairwise(self.subtitle_texts.items()), start=2):
            key1_name, key1_text, key2_name, key2_text = key1[0], key1[1], key2[0], key2[1]
            # print(index, no_of_keys, key1_name, key1_text, key2_name, key2_text)
            if key1_text == key2_text and index != no_of_keys:
                if not starting_key:
                    starting_key = key1_name
            else:
                # print("Text not equal\n")
                if not starting_key:  # This condition is used when the key doesn't match the previous or next key.
                    starting_key = key1_name
                duration = f"{starting_key}{self.divider}{key1_name}"
                new_subtitle_dict[duration] = key1_text
                if index == no_of_keys:  # The last key is always added to end of dictionary to avoid being skipped.
                    # print("No of keys reached!")
                    duration = f"{key2_name}{self.divider}{key2_name}"
                    new_subtitle_dict[duration] = key2_text
                starting_key = None
        self.subtitle_texts = new_subtitle_dict

    @staticmethod
    def similarity(text1: str, text2: str) -> float:
        return SequenceMatcher(a=text1, b=text2).quick_ratio()

    def similar_text_name_gen(self, start_name: str, end_name: str) -> str:
        """
        Takes 2 name durations and creates a new name.
        """
        start_name, end_name = start_name.split(self.divider)[0], end_name.split(self.divider)[1]
        new_name = f"{start_name}{self.divider}{end_name}"
        return new_name

    def name_to_duration(self, name: str) -> float:
        """
        Takes a name with two numbers and subtracts to get the duration in milliseconds.
        :param name: Name numbers should be separated by divider.
        :return: Duration
        """
        name_timecode = name.split(self.divider)
        duration = float(name_timecode[1]) - float(name_timecode[0])
        return duration

    def merge_adjacent_similar_texts(self) -> None:
        """
        Merge texts that are not the same but beside each other and similar.
        The text that has the longest duration becomes the text for all similar texts.
        """
        logger.debug("Merging adjacent similar texts")
        similarity_threshold = utils.Config.text_similarity_threshold  # Cut off point to determine similarity.
        new_subtitle_dict, no_of_keys = {}, len(self.subtitle_texts)
        starting_key = starting_key_txt = starting_key_dur = None
        for index, (key1, key2) in enumerate(pairwise(self.subtitle_texts.items()), start=2):
            key1_name, key1_txt, key1_dur = key1[0], key1[1], self.name_to_duration(key1[0])
            key2_name, key2_txt, key2_dur = key2[0], key2[1], self.name_to_duration(key2[0])
            similarity = self.similarity(key1_txt, key2_txt)
            # print(f"Index: {index}, No of Keys: {no_of_keys}\n"
            #       f"Key 1 Name: {key1_name}, Duration: {key1_dur}, Text: {key1_txt}\n"
            #       f"Key 2 Name: {key2_name}, Duration: {key2_dur}, Text: {key2_txt}\n"
            #       f"Key 1 & 2 Similarity: {similarity}")
            if similarity >= similarity_threshold and index != no_of_keys:
                if not starting_key:
                    starting_key, starting_key_txt, starting_key_dur = key1_name, key1_txt, key1_dur

                if key2_dur > starting_key_dur:  # Change text and duration when longer duration is found.
                    # print(f"--- Longer duration found: {key2_dur} ---")
                    starting_key_txt, starting_key_dur = key2_txt, key2_dur
            else:
                if not starting_key:  # This condition is used when the key doesn't match the previous or next key.
                    starting_key, starting_key_txt = key1_name, key1_txt

                if index == no_of_keys:
                    # print("No of keys reached!")
                    ending_key = key2_name  # This doesn't work well when the last key's text is not similar.
                else:
                    ending_key = key1_name

                new_key_name = self.similar_text_name_gen(starting_key, ending_key)
                # print(f"New key name: {new_key_name} \nNew key text: {starting_key_txt}\n")
                new_subtitle_dict[new_key_name] = starting_key_txt
                starting_key = starting_key_txt = starting_key_dur = None
        self.subtitle_texts = new_subtitle_dict

    def delete_keys(self, keys: set) -> None:
        """
        Delete all key durations in the set if they exist.
        """
        for key in keys:
            if key in self.subtitle_texts:
                del self.subtitle_texts[key]

    def remove_short_duration_consecutive_subs(self) -> None:
        """
        Deletes keys that contain subtitles that have durations that are shorter than the given minimum duration
        in the given number of consecutive rows.
        """
        logger.debug("Removing short duration consecutive subs")
        # Minimum allowed consecutive duration in milliseconds.
        min_consecutive_sub_dur = utils.Config.min_consecutive_sub_dur_ms
        # Maximum allowed number of short durations in a row.
        max_consecutive_short_durs = utils.Config.max_consecutive_short_durs

        keys_for_deletion, short_dur_keys, no_of_keys = set(), set(), len(self.subtitle_texts)
        for index, (dur_1, dur_2) in enumerate(pairwise(self.subtitle_texts), start=2):
            key1_dur, key2_dur = self.name_to_duration(dur_1), self.name_to_duration(dur_2)
            # print(f"Index: {index}, No of Keys: {no_of_keys}\n"
            #       f"Key 1 Name: {dur_1}, Duration: {key1_dur}\n"
            #       f"Key 2 Name: {dur_2}, Duration: {key2_dur}")
            if key1_dur < min_consecutive_sub_dur and key2_dur < min_consecutive_sub_dur and index != no_of_keys:
                short_dur_keys.add(dur_1)
                short_dur_keys.add(dur_2)
            else:
                if len(short_dur_keys) >= max_consecutive_short_durs:
                    # print(f"Short durations found for deletion! Keys: ({len(short_dur_keys)}) = {short_dur_keys}\n")
                    keys_for_deletion.update(short_dur_keys)
                short_dur_keys = set()
        self.delete_keys(keys_for_deletion)

    def remove_short_duration_subs(self) -> None:
        """
        Deletes keys that contain subtitles that have durations that are shorter than the minimum duration.
        """
        logger.debug("Removing short duration subs")
        # Minimum allowed time in milliseconds.
        min_sub_duration = utils.Config.min_sub_duration_ms
        short_dur_keys = set()
        for ms_duration in self.subtitle_texts:
            duration = self.name_to_duration(ms_duration)
            if duration <= min_sub_duration:
                short_dur_keys.add(ms_duration)
        self.delete_keys(short_dur_keys)

    def process_extracted_texts(self) -> None:
        """
        Process extracted texts in dictionary.
        """
        logger.debug("Processing extracted texts...")
        self.merge_adjacent_equal_texts()
        self.merge_adjacent_similar_texts()
        self.remove_short_duration_consecutive_subs()
        self.remove_short_duration_subs()

    @staticmethod
    def timecode(frame_no_in_milliseconds: float) -> str:
        """
        Use to frame no in milliseconds to create timecode.
        """
        # Calculate the components of the timecode.
        total_seconds = frame_no_in_milliseconds // 1000  # Convert milliseconds to total seconds.
        milliseconds_remainder = frame_no_in_milliseconds % 1000  # Calculate the remaining milliseconds.
        seconds = total_seconds % 60  # Calculate the seconds component (remainder after removing minutes).
        minutes = (total_seconds // 60) % 60
        hours = total_seconds // 3600  # Calculate the number of hours in the total seconds.
        return "%02d:%02d:%02d,%03d" % (hours, minutes, seconds, milliseconds_remainder)

    def generate_subtitle(self) -> list:
        """
        Use processed text files in dictionary to create subtitle file.
        """
        # Cancel if process has been cancelled by gui.
        if utils.Process.interrupt_process:
            logger.warning("Subtitle generation process interrupted!")
            return []

        logger.info("Generating subtitle...")
        subtitles = []
        for line_code, (ms_dur, txt) in enumerate(self.subtitle_texts.items(), start=1):
            key_name = ms_dur.split(self.divider)
            frame_start, frame_end = self.timecode(float(key_name[0])), self.timecode(float(key_name[1]))
            subtitle_line = f"{line_code}\n{frame_start} --> {frame_end}\n{txt}\n\n"
            subtitles.append(subtitle_line)
        logger.info("Subtitle generated!")
        return subtitles

    def load_extracted_texts(self) -> None:
        """
        Load extracted texts files into dictionary. The name of the file which represents the duration in milliseconds
        will be the key and text of the file will be the value.
        The files will be sorted before being added to the dict, this prevents the need for sorting again.
        """
        logger.debug("Loading extracted tests...")
        for file in sorted(self.text_output.iterdir(), key=lambda name: float(name.stem)):
            file_text = file.read_text(encoding="utf-8")
            self.subtitle_texts[file.stem] = file_text

    def gen_sub_file_name(self) -> Path:
        """
        If the file name doesn't exist, return it directly.
        If the file name already exists, append a unique identifier to the file name.
        :return: new file name with path.
        """
        name = self.video_path.with_suffix(".srt")
        if not name.exists():
            return name
        else:
            suffix = 1  # Find an available unique name by appending a number.
            while True:
                new_file_path = Path(f"{name.parent}/{name.stem} ({suffix}).srt")
                if not new_file_path.exists():
                    return new_file_path
                suffix += 1

    def save_subtitle(self, lines: list) -> None:
        """
        Save generated subtitle file in the same location as video file.
        :param lines: subtitle lines to be written to file.
        """
        if not lines:
            logger.debug(f"No lines in subtitles generated. Name: {self.video_path.name}")
            return
        name = self.gen_sub_file_name()
        with open(name, 'w', encoding="utf-8") as new_sub:
            new_sub.writelines(lines)
        logger.info(f"Subtitle file saved. Name: {name}")

    def get_frames_and_texts(self, sub_area: tuple, start_frame: int | None, stop_frame: int | None) -> None:
        """
        Get the frames and the images from the video by calling external functions.
        """
        video_to_frames(str(self.video_path), self.frame_output, sub_area, start_frame, stop_frame)
        frames_to_text(self.frame_output, self.text_output)

    def run_extraction(self, video_path: str, sub_area: tuple = None, start_frame: int = None,
                       stop_frame: int = None) -> None:
        """
        Run through the steps of extracting texts from subtitle area in video to create subtitle.
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists() or not self.video_path.is_file():
            logger.error(f"Video file: {self.video_path.name} ...could not be found!\n")
            return
        start = cv.getTickCount()
        self._empty_cache()  # Empty cache at the beginning of program run before it recreates itself.
        # If the directories do not exist, create the directories.
        self.frame_output.mkdir(parents=True)
        self.text_output.mkdir(parents=True)

        fps, frame_total, frame_width, frame_height = self.video_details(video_path)
        sub_area = sub_area or self.default_sub_area(frame_width, frame_height)

        logger.info(f"File Path: {self.video_path}\n"
                    f"Frame Total: {frame_total}, Frame Rate: {fps}\n"
                    f"Resolution: {frame_width} X {frame_height}\n"
                    f"Subtitle Area: {sub_area}\n"
                    f"Start Frame No: {start_frame}, Stop Frame No: {stop_frame}")

        self.get_frames_and_texts(sub_area, start_frame, stop_frame)
        self.load_extracted_texts()
        self.process_extracted_texts()
        subtitles = self.generate_subtitle()
        self.save_subtitle(subtitles)

        end = cv.getTickCount()
        total_time = (end - start) / cv.getTickFrequency()
        converted_time = time.strftime("%Hh:%Mm:%Ss", time.gmtime(total_time))
        logger.info(f"Subtitle Extraction Done! Total time: {converted_time}\n")
        self._empty_cache()
