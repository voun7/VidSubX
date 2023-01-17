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
    config_file = Path("utilities/config.ini")
    config = ConfigParser()
    config.read(config_file)

    sections = ["Frame Extraction", "Text Extraction", "Subtitle Generator"]
    keys = ["frame_extraction_frequency", "frame_extraction_chunk_size", "text_extraction_chunk_size",
            "ocr_max_processes", "ocr_rec_language", "text_similarity_threshold"]

    # Initial values
    frame_extraction_frequency = frame_extraction_chunk_size = None
    text_extraction_chunk_size = ocr_max_processes = ocr_rec_language = None
    text_similarity_threshold = None

    def __init__(self) -> None:
        if not self.config_file.exists():
            self.create_config_file()
        self.load_config()

    @classmethod
    def load_config(cls) -> None:
        cls.frame_extraction_frequency = cls.config[cls.sections[0]][cls.keys[0]]
        cls.frame_extraction_chunk_size = cls.config[cls.sections[0]][cls.keys[1]]
        cls.text_extraction_chunk_size = cls.config[cls.sections[1]][cls.keys[2]]
        cls.ocr_max_processes = cls.config[cls.sections[1]][cls.keys[3]]
        cls.ocr_rec_language = cls.config[cls.sections[1]][cls.keys[4]]
        cls.text_similarity_threshold = cls.config[cls.sections[2]][cls.keys[5]]

    def create_config_file(self) -> None:
        self.config[self.sections[0]] = {self.keys[0]: "2",
                                         self.keys[1]: "250"}
        self.config[self.sections[1]] = {self.keys[2]: "150",
                                         self.keys[3]: "4",
                                         self.keys[4]: "ch"}
        self.config[self.sections[2]] = {self.keys[5]: "0.65"}
        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)

    @classmethod
    def set_frame_extraction_frequency(cls, frequency: int) -> None:
        cls.frame_extraction_frequency = frequency
        cls.config[cls.sections[0]][cls.keys[0]] = str(cls.frame_extraction_frequency)
        logger.debug(f"{cls.keys[0]} set to: {cls.frame_extraction_frequency}")

    @classmethod
    def set_frame_extraction_chunk_size(cls, chunk_size: int) -> None:
        cls.frame_extraction_chunk_size = chunk_size
        cls.config[cls.sections[0]][cls.keys[1]] = str(cls.frame_extraction_chunk_size)
        logger.debug(f"{cls.keys[1]} set to: {cls.frame_extraction_chunk_size}")

    @classmethod
    def set_text_extraction_chunk_size(cls, chunk_size: int) -> None:
        cls.text_extraction_chunk_size = chunk_size
        cls.config[cls.sections[1]][cls.keys[2]] = str(cls.text_extraction_chunk_size)
        logger.debug(f"{cls.keys[2]} set to: {cls.text_extraction_chunk_size}")

    @classmethod
    def set_ocr_max_processes(cls, max_processes: int) -> None:
        cls.ocr_max_processes = max_processes
        cls.config[cls.sections[1]][cls.keys[3]] = str(cls.ocr_max_processes)
        logger.debug(f"{cls.keys[3]} set to: {cls.ocr_max_processes}")

    @classmethod
    def set_ocr_rec_language(cls, language: str) -> None:
        cls.ocr_rec_language = language
        cls.config[cls.sections[1]][cls.keys[4]] = cls.ocr_rec_language
        logger.debug(f"{cls.keys[4]} set to: {cls.ocr_rec_language}")

    @classmethod
    def set_text_similarity_threshold(cls, threshold: float) -> None:
        cls.text_similarity_threshold = threshold
        cls.config[cls.sections[2]][cls.keys[5]] = str(cls.text_similarity_threshold)
        logger.debug(f"{cls.keys[5]} set to: {cls.text_similarity_threshold}")


def print_progress(iteration, total, prefix='', suffix='', decimals=3, bar_length=25):
    """
    Call in a loop to create standard out progress bar
    :param iteration: current iteration
    :param total: total iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    :return: None
    """

    format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
    percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
    filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
    bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
    # print(f"\r{prefix} |{bar}| {percents}% {suffix}", end='', flush=True)  # prints progress on the same line
    logger.info(f"{prefix} |{bar}| {percents}% {suffix}")


if __name__ == '__main__':
    pass
else:
    Config()
