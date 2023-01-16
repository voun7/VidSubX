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

    # Initial values
    frame_extraction_frequency = None
    frame_extraction_chunk_size = None
    text_extraction_chunk_size = None
    ocr_max_processes = None
    ocr_rec_language = None
    text_similarity_threshold = None

    def __init__(self) -> None:
        if not self.config_file.exists():
            self.create_config_file()
        self.load_config()

    @classmethod
    def load_config(cls) -> None:
        cls.frame_extraction_frequency = cls.config["Frame extraction"]["frame_extraction_frequency"]
        cls.frame_extraction_chunk_size = cls.config["Frame extraction"]["frame_extraction_chunk_size"]
        cls.text_extraction_chunk_size = cls.config["Text extraction"]["text_extraction_chunk_size"]
        cls.ocr_max_processes = cls.config["Text extraction"]["ocr_max_processes"]
        cls.ocr_rec_language = cls.config["Text extraction"]["ocr_rec_language"]
        cls.text_similarity_threshold = cls.config["Subtitle generator"]["text_similarity_threshold"]

    def create_config_file(self) -> None:
        self.config["Frame extraction"] = {"frame_extraction_frequency": "2",
                                           "frame_extraction_chunk_size": "250"}
        self.config["Text extraction"] = {"text_extraction_chunk_size": "150",
                                          "ocr_max_processes": "4",
                                          "ocr_rec_language": "ch"}
        self.config["Subtitle generator"] = {"text_similarity_threshold": "0.65"}
        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)

    @classmethod
    def set_frame_extraction_frequency(cls, frequency: int) -> None:
        cls.frame_extraction_frequency = frequency
        cls.config["Frame extraction"]["frame_extraction_frequency"] = str(cls.frame_extraction_frequency)
        logger.debug(f"frame_extraction_frequency set to: {cls.frame_extraction_frequency}")

    @classmethod
    def set_frame_extraction_chunk_size(cls, chunk_size: int) -> None:
        cls.frame_extraction_chunk_size = chunk_size
        cls.config["Frame extraction"]["frame_extraction_chunk_size"] = str(cls.frame_extraction_chunk_size)
        logger.debug(f"frame_extraction_chunk_size set to: {cls.frame_extraction_chunk_size}")

    @classmethod
    def set_text_extraction_chunk_size(cls, chunk_size: int) -> None:
        cls.text_extraction_chunk_size = chunk_size
        cls.config["Text extraction"]["text_extraction_chunk_size"] = str(cls.text_extraction_chunk_size)
        logger.debug(f"text_extraction_chunk_size set to: {cls.text_extraction_chunk_size}")

    @classmethod
    def set_ocr_max_processes(cls, max_processes: int) -> None:
        cls.ocr_max_processes = max_processes
        cls.config["Text extraction"]["ocr_max_processes"] = str(cls.ocr_max_processes)
        logger.debug(f"ocr_max_processes set to: {cls.ocr_max_processes}")

    @classmethod
    def set_ocr_rec_language(cls, language: str) -> None:
        cls.ocr_rec_language = language
        cls.config["Text extraction"]["ocr_rec_language"] = cls.ocr_rec_language
        logger.debug(f"ocr_rec_language set to: {cls.ocr_rec_language}")

    @classmethod
    def set_text_similarity_threshold(cls, threshold: float) -> None:
        cls.text_similarity_threshold = threshold
        cls.config["Subtitle generator"]["text_similarity_threshold"] = str(cls.text_similarity_threshold)
        logger.debug(f"text_similarity_threshold set to: {cls.text_similarity_threshold}")


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
