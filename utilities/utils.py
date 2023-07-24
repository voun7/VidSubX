import logging
from configparser import ConfigParser
from pathlib import Path

logger = logging.getLogger(__name__)


class Process:
    interrupt_process = False

    @classmethod
    def start_process(cls) -> None:
        """
        Allows process to run.
        """
        cls.interrupt_process = False
        logger.debug(f"interrupt_process set to: {cls.interrupt_process}")

    @classmethod
    def stop_process(cls) -> None:
        """
        Stops process from running.
        """
        cls.interrupt_process = True
        logger.debug(f"interrupt_process set to: {cls.interrupt_process}")


class Config:
    # Config file location will always be the same regardless of which module starts the program.
    config_file = Path(__file__).parent.parent / "config.ini"
    config = ConfigParser()
    config.read(config_file)

    sections = ["Frame Extraction", "Text Extraction", "Subtitle Generator", "Subtitle Detection", "Notification"]
    keys = ["frame_extraction_frequency", "frame_extraction_chunk_size", "text_extraction_chunk_size",
            "ocr_max_processes", "ocr_rec_language", "text_similarity_threshold", "min_consecutive_sub_dur_ms",
            "max_consecutive_short_durs", "min_sub_duration_ms", "split_start", "split_stop", "no_of_frames",
            "sub_area_x_rel_padding", "sub_area_y_abs_padding", "use_search_area", "win_notify_sound",
            "win_notify_loop_sound"]

    # Permanent values
    subarea_height_scaler = 0.75

    # Default values
    default_frame_extraction_frequency = 2
    default_frame_extraction_chunk_size = 250

    default_text_extraction_chunk_size = 150
    default_ocr_max_processes = 4
    default_ocr_rec_language = "ch"

    default_text_similarity_threshold = 0.65
    default_min_consecutive_sub_dur_ms = 500.0
    default_max_consecutive_short_durs = 4
    default_min_sub_duration_ms = 120.0

    default_split_start = 0.25
    default_split_stop = 0.5
    default_no_of_frames = 200
    default_sub_area_x_rel_padding = 0.85
    default_sub_area_y_abs_padding = 10
    default_use_search_area = True

    default_win_notify_sound = "Default"
    default_win_notify_loop_sound = True

    # Initial values
    frame_extraction_frequency = frame_extraction_chunk_size = None
    text_extraction_chunk_size = ocr_max_processes = ocr_rec_language = None
    text_similarity_threshold = min_consecutive_sub_dur_ms = max_consecutive_short_durs = min_sub_duration_ms = None
    split_start = split_stop = no_of_frames = sub_area_x_rel_padding = sub_area_y_abs_padding = use_search_area = None
    win_notify_sound = win_notify_loop_sound = None

    def __init__(self) -> None:
        if not self.config_file.exists():
            self.create_default_config_file()
        self.load_config()

    def create_default_config_file(self) -> None:
        """
        Creates a new config file with the default values.
        """
        self.config[self.sections[0]] = {self.keys[0]: str(self.default_frame_extraction_frequency),
                                         self.keys[1]: self.default_frame_extraction_chunk_size}
        self.config[self.sections[1]] = {self.keys[2]: self.default_text_extraction_chunk_size,
                                         self.keys[3]: self.default_ocr_max_processes,
                                         self.keys[4]: self.default_ocr_rec_language}
        self.config[self.sections[2]] = {self.keys[5]: str(self.default_text_similarity_threshold),
                                         self.keys[6]: self.default_min_consecutive_sub_dur_ms,
                                         self.keys[7]: self.default_max_consecutive_short_durs,
                                         self.keys[8]: self.default_min_sub_duration_ms}
        self.config[self.sections[3]] = {self.keys[9]: str(self.default_split_start),
                                         self.keys[10]: self.default_split_stop,
                                         self.keys[11]: self.default_no_of_frames,
                                         self.keys[12]: self.default_sub_area_x_rel_padding,
                                         self.keys[13]: self.default_sub_area_y_abs_padding,
                                         self.keys[14]: self.default_use_search_area}
        self.config[self.sections[4]] = {self.keys[15]: self.default_win_notify_sound,
                                         self.keys[16]: self.default_win_notify_loop_sound}
        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)

    @classmethod
    def load_config(cls) -> None:
        """
        Parse the values of the config file into memory.
        """
        cls.frame_extraction_frequency = cls.config[cls.sections[0]].getint(cls.keys[0])
        cls.frame_extraction_chunk_size = cls.config[cls.sections[0]].getint(cls.keys[1])

        cls.text_extraction_chunk_size = cls.config[cls.sections[1]].getint(cls.keys[2])
        cls.ocr_max_processes = cls.config[cls.sections[1]].getint(cls.keys[3])
        cls.ocr_rec_language = cls.config[cls.sections[1]][cls.keys[4]]

        cls.text_similarity_threshold = cls.config[cls.sections[2]].getfloat(cls.keys[5])
        cls.min_consecutive_sub_dur_ms = cls.config[cls.sections[2]].getfloat(cls.keys[6])
        cls.max_consecutive_short_durs = cls.config[cls.sections[2]].getint(cls.keys[7])
        cls.min_sub_duration_ms = cls.config[cls.sections[2]].getfloat(cls.keys[8])

        cls.split_start = cls.config[cls.sections[3]].getfloat(cls.keys[9])
        cls.split_stop = cls.config[cls.sections[3]].getfloat(cls.keys[10])
        cls.no_of_frames = cls.config[cls.sections[3]].getint(cls.keys[11])
        cls.sub_area_x_rel_padding = cls.config[cls.sections[3]].getfloat(cls.keys[12])
        cls.sub_area_y_abs_padding = cls.config[cls.sections[3]].getint(cls.keys[13])
        cls.use_search_area = cls.config[cls.sections[3]].getboolean(cls.keys[14])

        cls.win_notify_sound = cls.config[cls.sections[4]][cls.keys[15]]
        cls.win_notify_loop_sound = cls.config[cls.sections[4]].getboolean(cls.keys[16])

    @classmethod
    def set_config(cls, **kwargs: int | float | str | bool) -> None:
        """
        Write new configuration values into memory & file.
        Config values must be strings.
        """
        # Write the new config value into the class variable (memory).
        cls.frame_extraction_frequency = kwargs.get(cls.keys[0], cls.frame_extraction_frequency)
        # Write the value of the class variable into to the config parser (file).
        cls.config[cls.sections[0]][cls.keys[0]] = str(cls.frame_extraction_frequency)
        cls.frame_extraction_chunk_size = kwargs.get(cls.keys[1], cls.frame_extraction_chunk_size)
        cls.config[cls.sections[0]][cls.keys[1]] = str(cls.frame_extraction_chunk_size)

        cls.text_extraction_chunk_size = kwargs.get(cls.keys[2], cls.text_extraction_chunk_size)
        cls.config[cls.sections[1]][cls.keys[2]] = str(cls.text_extraction_chunk_size)
        cls.ocr_max_processes = kwargs.get(cls.keys[3], cls.ocr_max_processes)
        cls.config[cls.sections[1]][cls.keys[3]] = str(cls.ocr_max_processes)
        cls.ocr_rec_language = kwargs.get(cls.keys[4], cls.ocr_rec_language)
        cls.config[cls.sections[1]][cls.keys[4]] = cls.ocr_rec_language

        cls.text_similarity_threshold = kwargs.get(cls.keys[5], cls.text_similarity_threshold)
        cls.config[cls.sections[2]][cls.keys[5]] = str(cls.text_similarity_threshold)
        cls.min_consecutive_sub_dur_ms = kwargs.get(cls.keys[6], cls.min_consecutive_sub_dur_ms)
        cls.config[cls.sections[2]][cls.keys[6]] = str(cls.min_consecutive_sub_dur_ms)
        cls.max_consecutive_short_durs = kwargs.get(cls.keys[7], cls.max_consecutive_short_durs)
        cls.config[cls.sections[2]][cls.keys[7]] = str(cls.max_consecutive_short_durs)
        cls.min_sub_duration_ms = kwargs.get(cls.keys[8], cls.min_sub_duration_ms)
        cls.config[cls.sections[2]][cls.keys[8]] = str(cls.min_sub_duration_ms)

        cls.split_start = kwargs.get(cls.keys[9], cls.split_start)
        cls.config[cls.sections[3]][cls.keys[9]] = str(cls.split_start)
        cls.split_stop = kwargs.get(cls.keys[10], cls.split_stop)
        cls.config[cls.sections[3]][cls.keys[10]] = str(cls.split_stop)
        cls.no_of_frames = kwargs.get(cls.keys[11], cls.no_of_frames)
        cls.config[cls.sections[3]][cls.keys[11]] = str(cls.no_of_frames)
        cls.sub_area_x_rel_padding = kwargs.get(cls.keys[12], cls.sub_area_x_rel_padding)
        cls.config[cls.sections[3]][cls.keys[12]] = str(cls.sub_area_x_rel_padding)
        cls.sub_area_y_abs_padding = kwargs.get(cls.keys[13], cls.sub_area_y_abs_padding)
        cls.config[cls.sections[3]][cls.keys[13]] = str(cls.sub_area_y_abs_padding)
        cls.use_search_area = kwargs.get(cls.keys[14], cls.use_search_area)
        cls.config[cls.sections[3]][cls.keys[14]] = str(cls.use_search_area)

        cls.win_notify_sound = kwargs.get(cls.keys[15], cls.win_notify_sound)
        cls.config[cls.sections[4]][cls.keys[15]] = cls.win_notify_sound
        cls.win_notify_loop_sound = kwargs.get(cls.keys[16], cls.win_notify_loop_sound)
        cls.config[cls.sections[4]][cls.keys[16]] = str(cls.win_notify_loop_sound)

        with open(cls.config_file, 'w') as configfile:
            cls.config.write(configfile)
        logger.debug("Configuration values changed!")


def print_progress(iteration: int, total: int, prefix: str = '', suffix: str = 'Complete', decimals: int = 3,
                   bar_length: int = 25) -> None:
    """
    Call in a loop to create standard out progress bar.
    :param iteration: current iteration
    :param total: total iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    """
    if not total:  # prevent error if total is zero.
        return

    format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
    percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
    filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
    bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
    print(f"\r{prefix} |{bar}| {percents}% {suffix}", end='', flush=True)  # prints progress on the same line

    if "100.0" in percents:  # prevent next line from joining previous line
        print()


if __name__ == '__main__':
    pass
else:
    Config()
