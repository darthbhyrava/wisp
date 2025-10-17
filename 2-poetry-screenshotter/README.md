# poetry-screenshotter

A specialized Python package for converting HTML files to high-quality PNG screenshots with precise targeting of selected content elements. In the current code, the HTML files are from Poetry Foundation, and the target element is the poem content within `div.poem-body`.

## Overview

Poetry Screenshotter is a browser automation tool designed specifically for capturing visual representations of poems from HTML files. The package leverages Playwright's browser automation capabilities to render HTML content and extract targeted elements (primarily targeting `.poem-body` CSS selectors) into PNG images.

### Package Structure
```
poetry_screenshotter/
├── core.py          # Main Screenshotter class with browser automation
├── cli.py           # Command-line interface with batch processing
├── api.py           # FastAPI REST endpoints
├── exceptions.py    # Custom error handling classes
└── poems/           # One parsed (not original) Poetry Foundation HTML file
```

The poem we use as an example here is [*"An Ode to Ben Jonson"*](https://www.poetryfoundation.org/poems/47331/an-ode-to-ben-jonson), written by [Robert Herrick](https://www.poetryfoundation.org/poets/robert-herrick) (1591—1674) for his literary "father". 


### Installation Requirements

#### Local Development Installation
```bash
# clone the repository and navigate to the project directory
$ cd 2-poetry-screenshotter

# create and activate a virtual environment
$ python3 -m venv screenshotter-venv
$ source screenshotter-venv/bin/activate 

# upgrade pip to latest version (recommended)
$ pip install --upgrade pip

# install the package in editable/development mode
$ pip install -e .
```

#### Browser Dependencies
```bash
# install Playwright browsers (required)
$ playwright install
# or install a specific browser
$ playwright install chromium
```

### Constraints and Technical Specs

#### Viewport and Display Settings
- **Fixed Viewport**: 1920x1080 pixels for consistent cross-platform rendering
- **Background Color**: White (`background: white !important`)
- **Body Margins**: Removed (`margin: 0, padding: 0`)
- **Element Positioning**: Preserves original Poetry Foundation layout

#### CSS Selector Requirements
- **Primary Target**: `div.poem-body` (required - triggers `PoemNotFoundError` if missing)
- **Element Validation**: Non-empty text content verification
- **Visibility Check**: 10-second timeout for element to become visible
- **Scroll Positioning**: Auto-scroll to poem element with 500ms settle time

## Usage Examples

### Command Line Interface

#### Basic Conversion
```bash
$ poetry-screenshotter convert ./path-to-poem.html -o ./path-to-desired-output.png
```

#### Batch Processing with Configuration
```bash
$ poetry-screenshotter batch 
  --input-dir ./html_files 
  --output-dir ./screenshots 
  --browser firefox 
  --max-workers 4 
  --timeout 60
```

#### Server Mode
```bash
poetry-screenshotter serve 
  --host 0.0.0.0 
  --port 8000 
  --workers 4
```

### CLI Parameters
- `--browser {chromium,firefox,webkit}`: Browser engine selection
- `--headless/--no-headless`: Headless mode toggle
- `--timeout INTEGER`: Page load timeout in seconds
- `--max-workers INTEGER`: Parallel processing limit
- `--host TEXT`: API server bind address
- `--port INTEGER`: API server port number

### REST API Endpoints

#### Single File Conversion
```bash
curl -X POST "http://localhost:8000/convert" 
  -F "file=@poem.html" 
  -F "browser=chromium" 
  -F "timeout=30"
```

#### Batch Conversion
```bash
curl -X POST "http://localhost:8000/batch-convert" 
  -F "files=@poem1.html" 
  -F "files=@poem2.html" 
  -F "browser=firefox"
```

### API Response Format
```json
{
  "filename": "poem.png",
  "size": 157284,
  "content_type": "image/png",
  "processing_time": 2.34,
  "browser_used": "chromium"
}
```

## Browser Installation Notes

### Automatic Installation
```bash
# Install all browsers
playwright install

# System dependencies (Linux)
playwright install-deps
```

### Manual Browser Paths
```python
# Custom browser executable paths
BROWSER_PATHS = {
    'chromium': '/usr/bin/chromium',
    'firefox': '/usr/bin/firefox',
    'webkit': '/usr/bin/webkit'
}
```