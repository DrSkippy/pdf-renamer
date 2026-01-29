import logging
import re

def make_filename_safe(filename):
    """
    Sanitizes a filename by removing or replacing invalid characters to ensure
    filesystem compatibility. The function performs multiple transformations:
    converts whitespace sequences to single underscores, removes non-ASCII
    alphanumeric characters (except underscores, dots, and hyphens), consolidates
    multiple consecutive underscores into one, and strips leading/trailing
    underscores from the result.

    :param filename: The original filename string to be sanitized
    :type filename: str
    :return: A sanitized filename string safe for filesystem operations
    :rtype: str
    """
    logging.debug(f"Making filename safe: {filename}")
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
