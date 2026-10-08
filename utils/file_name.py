import logging
import re
import unicodedata


def make_filename_safe(filename):
    """
    Sanitizes a filename by removing or replacing invalid characters to ensure
    filesystem compatibility. The function performs multiple transformations:
    applies NFKC normalization (so ligatures such as "ﬁ" become "fi"), turns
    Unicode dashes into hyphens, converts whitespace sequences to single underscores, removes non-ASCII
    alphanumeric characters (except underscores, dots, and hyphens), consolidates
    multiple consecutive underscores into one, and strips leading/trailing
    underscores from the result.

    :param filename: The original filename string to be sanitized
    :type filename: str
    :return: A sanitized filename string safe for filesystem operations
    :rtype: str
    """
    logging.debug(f"Making filename safe: {filename}")
    # Decompose compatibility characters (ligatures, full-width forms) to ASCII equivalents
    filename = unicodedata.normalize("NFKC", filename)
    # Keep word breaks from en/em dashes and similar instead of deleting them
    filename = re.sub(r"[\u2010-\u2015\u2212]", "-", filename)
    # Replace all spaces with a single underscore first
    filename = re.sub(r"\s+", "_", filename)
    # Remove any characters that are not ASCII alphanumeric, underscores, or hyphens
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "", filename)
    # Replace multiple consecutive underscores with a single underscore
    filename = re.sub(r"__+", "_", filename)
    # Remove leading/trailing underscores
    filename = filename.strip("_")
    logging.info(f"Safe filename: {filename}")
    return filename
