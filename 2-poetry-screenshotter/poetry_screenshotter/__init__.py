"""
HTML to Image Screenshotter.

A package to convert HTML files to PNG images by screenshotting specific HTML content.
"""

__version__ = "0.1.0"
__author__ = "Sriharsh Bhyravajjula"
__email__ = ""

from .core import Screenshotter
from .exceptions import ScreenshotterError, HTMLParsingError, ScreenshotError

__all__ = [
    "Screenshotter",
    "ScreenshotterError",
    "HTMLParsingError",
    "ScreenshotError"
]