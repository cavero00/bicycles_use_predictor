import logging
from logging import Logger
import sys

def get_logger(name: str, log_file: str = "pipeline.log") -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Captura todo

    # Formatter con timestamp y run_id (se puede inyectar dinámicamente)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )

    # Handler consola
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Handler archivo
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
