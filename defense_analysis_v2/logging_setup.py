"""Logging setup used by every module in the package."""

import logging
import os
from pathlib import Path


def setup_logging(output_dir: str, name: str = "defense_analysis_v2",
                  level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes to both the console and ``analysis.log``
    inside ``output_dir``.

    Safe to call repeatedly with the same name; existing handlers are not
    duplicated.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called twice
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    fh = logging.FileHandler(Path(output_dir) / "analysis.log")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Don't propagate to root (avoids double-logging if a library configured root)
    logger.propagate = False

    return logger
