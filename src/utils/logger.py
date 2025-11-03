import logging
import sys
import os
from logging.handlers import RotatingFileHandler

# Define the logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"


def setup_logger(
    name="neurosymbolic_vqa",
    log_level=logging.INFO,
    log_dir="logs",
    log_filename="app.log",
):
    """
    Configures and returns a logger with both console and file handlers.

    Args:
        name (str): The name of the logger.
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        log_dir (str): The directory to store log files.
        log_filename (str): The name of the log file.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)

    # Get the logger instance
    logger = logging.getLogger(name)
    logger.setLevel(log_level)  # Set the logger's base level

    # Prevent logs from propagating to the root logger
    logger.propagate = False

    # Clear existing handlers to avoid duplicates (especially in notebooks)
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Console Handler ---
    # Logs to standard output (the console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- File Handler ---
    # Logs to a rotating file
    # Rotates when the file reaches 5MB, keeps 3 backup files
    file_handler = RotatingFileHandler(
        log_filepath, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info("Logger configured. Logging to console and %s", log_filepath)

    return logger


# A default logger instance to be imported by other modules
log = setup_logger()
