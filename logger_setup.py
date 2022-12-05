import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def get_logger() -> None:
    # Create folder for file logs
    log_dir = Path(f"{Path.cwd()}/logs")
    if not log_dir.exists():
        log_dir.mkdir()

    # Create a custom base_logger
    base_logger = logging.getLogger()
    base_logger.setLevel(logging.DEBUG)

    # Create handlers
    file_handler = TimedRotatingFileHandler(
        'logs/runtime.log', when='midnight', interval=1, backupCount=7, encoding='utf-8'
    )
    file_handler.namer = my_namer
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    main_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # The console sends only messages by default no need for formatter
    file_handler.setFormatter(main_format)

    # Add handlers to the base_logger
    base_logger.addHandler(file_handler)
    base_logger.addHandler(console_handler)


def my_namer(default_name):
    # This will be called when doing the log rotation
    # default_name is the default filename that would be assigned, e.g. Rotate_Test.txt.YYYY-MM-DD
    # Do any manipulations to that name here, for example this changes the name to Rotate_Test.YYYY-MM-DD.txt
    base_filename, ext, date = default_name.split(".")
    return f"{base_filename}.{date}.{ext}"


# # Use the following to add logger to other modules.
# from logger_setup import get_logger
# logger = get_logger(__name__)
# Do not log this messages unless they are at least warnings
# logging.getLogger("").setLevel(logging.WARNING)
