import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def get_console_handler() -> logging.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    # The console sends only messages by default no need for formatter
    return console_handler


def get_file_handler() -> logging.handlers:
    log_file = "./logs/runtime.log"
    file_handler = TimedRotatingFileHandler(log_file, when='midnight', interval=1, backupCount=7, encoding='utf-8')
    file_handler.namer = my_namer
    file_handler.setLevel(logging.DEBUG)
    file_log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_log_format)
    return file_handler


def get_logger():
    # Create folder for file logs
    log_dir = Path("./logs")
    if not log_dir.exists():
        log_dir.mkdir()

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # better to have too much log than not enough

    # Add handlers to the logger
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())


def my_namer(default_name):
    # This will be called when doing the log rotation
    # default_name is the default filename that would be assigned, e.g. Rotate_Test.txt.YYYY-MM-DD
    # Do any manipulations to that name here, for example this changes the name to Rotate_Test.YYYY-MM-DD.txt
    base_filename, ext, date = default_name.split(".")
    return f"{base_filename}.{date}.{ext}"

# # Use the following to add custom logger to other modules.
# import logging
# logger = logging.getLogger(__name__)
# Do not log this messages unless they are at least warnings
# logging.getLogger("").setLevel(logging.WARNING)

# Go to C:\Users\VOUN-XPS\miniconda3\envs\VSE\Lib\site-packages\paddle\distributed\utils\log_utils.py
# Change the get_logger function's logging.getLogger(name) to logging.getLogger(__name__)
# This prevents duplicate logs from paddle
