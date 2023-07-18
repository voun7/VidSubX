import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def get_console_handler() -> logging.handlers:
    """
    The console sends only messages by default no need for formatter.
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    return console_handler


def get_file_handler(log_path: Path, log_format: logging.Formatter) -> logging.handlers:
    log_file = log_path / "runtime.log"
    file_handler = TimedRotatingFileHandler(log_file, when='midnight', interval=1, backupCount=7, encoding='utf-8')
    file_handler.namer = my_namer
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    return file_handler


def get_logger():
    """
    Use the following to add logger to other modules.
    import logging
    logger = logging.getLogger(__name__)

    The following suppress log messages. It will not log messages of given module unless they are at least warnings.
    logging.getLogger("module_name").setLevel(logging.WARNING)
    """
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
    logger.addHandler(get_file_handler(log_dir, log_format))


def my_namer(default_name: str) -> str:
    """
    This will be called when doing the log rotation
    default_name is the default filename that would be assigned, e.g. Rotate_Test.txt.YYYY-MM-DD
    Do any manipulations to that name here, for example this function changes the name to Rotate_Test.YYYY-MM-DD.txt
    """
    base_filename, ext, date = default_name.split(".")
    return f"{base_filename}.{date}.{ext}"

# Go to C:\Users\VOUN-XPS\miniconda3\envs\VSE\Lib\site-packages\paddle\distributed\utils\log_utils.py
# Change the get_logger function's logging.getLogger(name) to logging.getLogger(__name__)
# This prevents duplicate logs from paddle
