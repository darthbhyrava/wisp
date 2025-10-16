import asyncio
import gzip
import logging
import os

from bs4 import BeautifulSoup
from pathlib import Path
from playwright.async_api import async_playwright, Page, Browser
from typing import Optional, Union

from .exceptions import HTMLParsingError, ScreenshotError, PoemNotFoundError


logger = logging.getLogger(__name__)


class Screenshotter:
    """Convert HTML files to PNG images."""

    def __init__(self, headless: bool = True, browser_type: str = "chromium"):
        """Initialize the screenshotter.
        Args:
            headless: Whether to run browser in headless mode
            browser_type: Type of browser to use ('chromium', 'firefox', 'webkit')
        """
        self.headless = headless
        self.browser_type = browser_type
        self._browser: Optional[Browser] = None
        self._playwright = None

    async def __aenter__(self):
        # use async context manager to properly handle browser lifecycle
        # since browser startup involves async I/O operations
        await self._start_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # ensure browser is properly closed with async cleanup
        await self._close_browser()

    async def _start_browser(self):
        """Start the playwright browser."""
        # use async because playwright.start() returns awaitable
        # and browser launch involves network/process I/O
        if self._playwright is None:
            self._playwright = await async_playwright().start()
        if self.browser_type == "chromium":
            browser_launcher = self._playwright.chromium
        elif self.browser_type == "firefox":
            browser_launcher = self._playwright.firefox
        elif self.browser_type == "webkit":
            browser_launcher = self._playwright.webkit
        else:
            raise ValueError(f"Unsupported browser type: {self.browser_type}")
        try:
            # try to use system browsers if playwright browsers are not installed
            if self.browser_type == "chromium":
                # try system chrome/chromium paths
                for chrome_path in ['/usr/bin/google-chrome', '/usr/bin/chromium-browser', '/usr/bin/chromium']:
                    if os.path.exists(chrome_path):
                        try:
                            # browser launch is async operation involving process creation
                            self._browser = await browser_launcher.launch(
                                headless=self.headless,
                                executable_path=chrome_path
                            )
                            logger.info(f"Started {self.browser_type} browser using {chrome_path} (headless={self.headless})")
                            return
                        except Exception:
                            continue
            # fallback to default playwright browser
            # await required for browser process startup
            self._browser = await browser_launcher.launch(headless=self.headless)
            logger.info(f"Started {self.browser_type} browser (headless={self.headless})")
        except Exception as e:
            raise ScreenshotError(
                f"Failed to start {self.browser_type} browser. "
                f"Please run 'playwright install {self.browser_type}' first. Error: {str(e)}"
            )

    async def _close_browser(self):
        # use async for proper cleanup of browser processes and connections
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    def _load_html_content(self, html_path: Union[str, Path]) -> str:
        """Path to HTML file (can be .html or .html.gz)"""
        html_path = Path(html_path)
        if not html_path.exists():
            raise HTMLParsingError(f"HTML file not found: {html_path}")
        try:
            if html_path.suffix == '.gz':
                with gzip.open(html_path, 'rt', encoding='utf-8') as f:
                    content = f.read()
            else:
                with open(html_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            logger.info(f"Loaded HTML content from {html_path} ({len(content)} characters)")
            return content
        except Exception as e:
            raise HTMLParsingError(f"Error reading HTML file {html_path}: {str(e)}")

    def _validate_poem_content(self, html_content: str) -> None:
        """Validate that the HTML contains poem content by checking for PF-specific 'poem-body' div."""
        soup = BeautifulSoup(html_content, 'html.parser')
        # locate the first <div> that has the "poem-body" class

        ##### NOTE: REPLACE THIS WITH ANY DIV CLASS YOU WANT TO TARGET #####
        poem_body = soup.find('div', class_='poem-body')
        ####################################################################

        if not poem_body:
            raise PoemNotFoundError("No element with 'poem-body' class found in HTML")
        poem_text = poem_body.get_text().strip()
        if not poem_text:
            raise PoemNotFoundError("Found poem-body element but it contains no text")
        logger.info(f"Found poem content with {len(poem_text)} characters")

    async def _screenshot_poem_element(
        self, 
        page: Page, 
        output_path: Union[str, Path],
        full_page: bool = False
    ) -> Path:
        """Screenshot the poem-body element from the page."""
        output_path = Path(output_path)
        try:
            # wait for page to load completely
            # use async to avoid blocking while network requests finish
            await page.wait_for_load_state("networkidle")
            # find the poem-body element
            poem_element = page.locator('div.poem-body').first
            # check if element exists
            # await required because count() queries the live DOM
            if await poem_element.count() == 0:
                raise ScreenshotError("poem-body element not found on rendered page")
            # wait for element to be visible
            # async wait prevents blocking while element renders
            await poem_element.wait_for(state="visible", timeout=10000)
            # scroll to the poem element to ensure we capture the very beginning
            # async operation that may trigger layout/paint cycles
            await poem_element.scroll_into_view_if_needed()
            # add a small delay for scrolling to settle
            # async timeout allows other operations to continue
            await page.wait_for_timeout(500)
            # take screenshot of the specific element with high quality settings
            # async because screenshot capture involves image processing and file I/O
            await poem_element.screenshot(
                path=str(output_path),
                type="png",
                omit_background=False  # keep background
            )
            logger.info(f"Screenshot saved to {output_path}")
            return output_path
        except Exception as e:
            raise ScreenshotError(f"Failed to take screenshot: {str(e)}")

    async def convert_to_image(
        self, 
        html_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None,
        full_page: bool = False
    ) -> Path:
        """Convert Poetry Foundation HTML file to PNG image."""
        html_path = Path(html_path)
        # generate output path if not provided
        if output_path is None:
            output_path = html_path.with_suffix('.png')
            if html_path.suffix == '.gz':
                # remove .gz and replace .html with .png
                output_path = html_path.with_suffix('').with_suffix('.png')
        else:
            output_path = Path(output_path)
        # ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # load and validate HTML content
        html_content = self._load_html_content(html_path)
        self._validate_poem_content(html_content)
        # start browser if not already started
        # async because browser startup involves process creation
        if self._browser is None:
            await self._start_browser()
        # Ensure browser is available (helps both runtime safety and static analysis)
        if self._browser is None:
            raise ScreenshotError("Browser failed to start; self._browser is still None")
        browser: Browser = self._browser
        # create a new page with specific viewport size for consistent rendering
        # using standard desktop resolution at 100% zoom equivalent
        # async because page creation involves browser communication
        page = await browser.new_page(viewport={'width': 1920, 'height': 1080})
        try:
            # set content and wait for it to load
            # async to avoid blocking while DOM parsing and resource loading occurs
            await page.set_content(html_content, wait_until="networkidle")
            # adding a tricky bit here: minimal CSS to hide only navigation elements that shouldn't be in screenshots
            # while preserving all poem formatting including percentage-based positioning
            # async because style injection triggers CSS parsing and layout recalculation
            await page.add_style_tag(content="""
                /* Hide navigation and header elements that might appear at top */
                nav, header, .header, .navbar, .nav, .navigation,
                .site-header, .page-header, .main-header, [class*="nav"], 
                [class*="menu"], .breadcrumb, .ads, .advertisement,
                .print\\:hidden {
                    display: none !important;
                }
                
                /* Ensure page doesn't have extra margins */
                body {
                    margin: 0 !important;
                    padding: 0 !important;
                    background: white !important;
                }
                
                /* Hide any fixed or absolute positioned elements that might interfere */
                [style*="position: fixed"], [style*="position: absolute"] {
                    display: none !important;
                }
                
                /* Ensure poem-body preserves original formatting - minimal changes */
                .poem-body {
                    background: white !important;
                    /* DO NOT override padding, margins, line-height, or white-space */
                    /* These are critical for preserving the poem's visual structure */
                }
            """)
            # give a moment for any dynamic content to render and CSS to apply
            # async timeout allows layout and paint cycles to complete without blocking
            await page.wait_for_timeout(1000)
            # take screenshot
            # async because screenshot process involves multiple async browser operations
            result_path = await self._screenshot_poem_element(page, output_path, full_page)
            logger.info(f"Successfully converted {html_path} to {result_path}")
            return result_path
        finally:
            # async close to properly cleanup page resources and event listeners
            await page.close()

    # synchronous wrapper method
    # provides sync interface while using async internally for browser operations
    def convert_to_image_sync(
        self, 
        html_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None,
        full_page: bool = False
    ) -> Path:
        """Synchronous wrapper for convert_to_image."""
        async def _run():
            async with self:
                return await self.convert_to_image(html_path, output_path, full_page)
        # use asyncio.run() to bridge sync/async gap for users who prefer sync interface
        return asyncio.run(_run())