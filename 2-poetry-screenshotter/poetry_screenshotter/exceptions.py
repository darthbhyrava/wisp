class ScreenshotterError(Exception):
    """Base exception for all Screenshotter errors."""
    pass


class HTMLParsingError(ScreenshotterError):
    """Raised when there's an error parsing the HTML file."""
    pass


class ScreenshotError(ScreenshotterError):
    """Raised when there's an error taking the screenshot."""
    pass


class PoemNotFoundError(HTMLParsingError):
    """Raised when the poem-body element is not found in the HTML."""
    pass