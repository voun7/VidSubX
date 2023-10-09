import io
import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


class LogLevelFilter(logging.Filter):
    def __init__(self, level: int) -> None:
        """
        Initialize the LogLevelFilter with the specified log level.
        :param level: The log level threshold. Log records with a level lower than this threshold will be allowed,
        while higher-level records will be filtered out.
        """
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        """
        This method is called by log handlers to decide whether to process the log record.
        If the log record's level is less than the filter's level, the record will be allowed,
        and it will be processed further. Otherwise, the record will be filtered out.
        :param record: The log record to be processed.
        :return: True if the log record should be processed, False otherwise.
        """
        # Comparing the log level of the record to the filter's level
        return record.levelno < self.level


def get_console_error_handler() -> logging.handlers:
    """
    Determine how stderr messages for the console will be handled.
    The console sends only messages by default no need for formatter.
    """
    error_handler = logging.StreamHandler()
    error_handler.setLevel(logging.ERROR)
    return error_handler


def get_console_handler() -> logging.handlers:
    """
    Determine how stdout messages for the console will be handled.
    The console sends only messages by default no need for formatter.
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(LogLevelFilter(logging.ERROR))
    return console_handler


def get_file_handler(log_dir: Path, log_format: logging.Formatter) -> logging.handlers:
    """
    Determine how the log messages are handled for log files.
    """
    log_file = log_dir / "runtime.log"
    file_handler = TimedRotatingFileHandler(log_file, when='midnight', interval=1, backupCount=7, encoding='utf-8')
    file_handler.namer = my_namer
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    return file_handler


def set_no_console_redirect() -> None:
    """
    When console is not available stdout and stderr values will be changed.
    """
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    if sys.stderr is None:
        sys.stderr = io.StringIO()


def reset_handlers() -> None:
    """
    Remove all handlers from the root logger.
    This helps prevent duplicate logs from other handlers created by imported modules.
    """
    logger = logging.getLogger()
    for handler in logger.handlers:
        logger.removeHandler(handler)


def setup_logging() -> None:
    """
    Use the following to add logger to other modules.
    import logging
    logger = logging.getLogger(__name__)

    The following suppress log messages. It will not log messages of given module unless they are at least warnings.
    logging.getLogger("module_name").setLevel(logging.WARNING)
    """
    # Run pre logging setup functions.
    reset_handlers()
    set_no_console_redirect()

    # Create folder for file logs.
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create a custom logger.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers.
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add handlers to the logger.
    logger.addHandler(get_console_handler())
    logger.addHandler(get_console_error_handler())
    logger.addHandler(get_file_handler(log_dir, log_format))


def my_namer(default_name: str) -> str:
    """
    This will be called when doing the log rotation
    default_name is the default filename that would be assigned, e.g. Rotate_Test.txt.YYYY-MM-DD
    Do any manipulations to that name here, for example this function changes the name to Rotate_Test.YYYY-MM-DD.txt
    """
    default_name = Path(default_name)
    file_path, ext, date = default_name.parent, default_name.suffixes[0], default_name.suffixes[1]
    base_filename = default_name.stem.replace(ext, "")
    return f"{file_path}/{base_filename}{date}{ext}"
