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
    config_file = Path("config.ini")
    config = ConfigParser()
    config.read(config_file)

    sections = ["Frame Extraction", "Text Extraction", "Subtitle Generator", "Subtitle Detection"]
    keys = ["frame_extraction_frequency", "frame_extraction_chunk_size", "text_extraction_chunk_size",
            "ocr_max_processes", "ocr_rec_language", "text_similarity_threshold", "split_start", "split_stop",
            "no_of_frames", "sub_area_x_padding", "sub_area_y_padding"]

    # Permanent values
    subarea_height_scaler = 0.75

    # Default values
    default_frame_extraction_frequency = 2
    default_frame_extraction_chunk_size = 250
    default_text_extraction_chunk_size = 150
    default_ocr_max_processes = 4
    default_ocr_rec_language = "ch"
    default_text_similarity_threshold = 0.65
    default_split_start = 0.25
    default_split_stop = 0.5
    default_no_of_frames = 200
    default_sub_area_x_padding = 200
    default_sub_area_y_padding = 10

    # Initial values
    frame_extraction_frequency = frame_extraction_chunk_size = None
    text_extraction_chunk_size = ocr_max_processes = ocr_rec_language = None
    text_similarity_threshold = None
    split_start = split_stop = no_of_frames = None
    sub_area_x_padding = sub_area_y_padding = None

    def __init__(self) -> None:
        if not self.config_file.exists():
            self.create_default_config_file()
        self.load_config()

    def create_default_config_file(self) -> None:
        self.config[self.sections[0]] = {self.keys[0]: str(self.default_frame_extraction_frequency),
                                         self.keys[1]: self.default_frame_extraction_chunk_size}
        self.config[self.sections[1]] = {self.keys[2]: self.default_text_extraction_chunk_size,
                                         self.keys[3]: self.default_ocr_max_processes,
                                         self.keys[4]: self.default_ocr_rec_language}
        self.config[self.sections[2]] = {self.keys[5]: str(self.default_text_similarity_threshold)}
        self.config[self.sections[3]] = {self.keys[6]: str(self.default_split_start),
                                         self.keys[7]: self.default_split_stop,
                                         self.keys[8]: self.default_no_of_frames,
                                         self.keys[9]: self.default_sub_area_x_padding,
                                         self.keys[10]: self.default_sub_area_y_padding}
        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)

    @classmethod
    def load_config(cls) -> None:
        cls.frame_extraction_frequency = int(cls.config[cls.sections[0]][cls.keys[0]])
        cls.frame_extraction_chunk_size = int(cls.config[cls.sections[0]][cls.keys[1]])
        cls.text_extraction_chunk_size = int(cls.config[cls.sections[1]][cls.keys[2]])
        cls.ocr_max_processes = int(cls.config[cls.sections[1]][cls.keys[3]])
        cls.ocr_rec_language = cls.config[cls.sections[1]][cls.keys[4]]
        cls.text_similarity_threshold = float(cls.config[cls.sections[2]][cls.keys[5]])
        cls.split_start = float(cls.config[cls.sections[3]][cls.keys[6]])
        cls.split_stop = float(cls.config[cls.sections[3]][cls.keys[7]])
        cls.no_of_frames = int(cls.config[cls.sections[3]][cls.keys[8]])
        cls.sub_area_x_padding = int(cls.config[cls.sections[3]][cls.keys[9]])
        cls.sub_area_y_padding = int(cls.config[cls.sections[3]][cls.keys[10]])

    @classmethod
    def set_config(cls, **kwargs: int | float | str) -> None:
        # Write into memory & file
        cls.frame_extraction_frequency = kwargs.get(cls.keys[0], cls.frame_extraction_frequency)
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

        cls.split_start = kwargs.get(cls.keys[6], cls.split_start)
        cls.config[cls.sections[3]][cls.keys[6]] = str(cls.split_start)
        cls.split_stop = kwargs.get(cls.keys[7], cls.split_stop)
        cls.config[cls.sections[3]][cls.keys[7]] = str(cls.split_stop)
        cls.no_of_frames = kwargs.get(cls.keys[8], cls.no_of_frames)
        cls.config[cls.sections[3]][cls.keys[8]] = str(cls.no_of_frames)
        cls.sub_area_x_padding = kwargs.get(cls.keys[9], cls.sub_area_x_padding)
        cls.config[cls.sections[3]][cls.keys[9]] = str(cls.sub_area_x_padding)
        cls.sub_area_y_padding = kwargs.get(cls.keys[10], cls.sub_area_y_padding)
        cls.config[cls.sections[3]][cls.keys[10]] = str(cls.sub_area_y_padding)

        with open(cls.config_file, 'w') as configfile:
            cls.config.write(configfile)
        logger.debug("Configuration values changed!")


def print_progress(iteration: int, total: int, prefix: str = '', suffix: str = '', decimals: int = 3,
                   bar_length: int = 25) -> None:
    """
    Call in a loop to create standard out progress bar
    :param iteration: current iteration
    :param total: total iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    """

    format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
    percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
    filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
    bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
    # print(f"\r{prefix} |{bar}| {percents}% {suffix}", end='', flush=True)  # prints progress on the same line
    print(f"{prefix} |{bar}| {percents}% {suffix}")


if __name__ == '__main__':
    pass
else:
    Config()
