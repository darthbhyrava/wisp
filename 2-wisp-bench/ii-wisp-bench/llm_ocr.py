import argparse
import base64
import mimetypes
import os
import random
import time

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

# optional imports - imported as needed based on model selection
try:
    import google.generativeai as genai
    from PIL import Image
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# default configs
DEFAULT_INPUT_DIR = "../poem_images/"
DEFAULT_OUTPUT_DIR = "./poem_texts/"
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]

# unified system prompt for all models
SYSTEM_PROMPT = """
## Objective:
Convert the poem image into plain text with exact preservation of its visual layout (spacing, alignment, and line breaks). Prioritize fidelity to the image structure and visual layout over standard formatting. Your task is purely transcription with layout preservation. Do not interpret, explain, or modify the text.


## Formatting Guidelines:
 Here are some guidelines to help with edge cases:
 - Use □ for unreadable characters
 - Ignore all typographical formatting like *italics*, **bold**, `underline`, or strikethrough. Transcribe only the text and its spacing.
 - **DO NOT** auto-wrap long lines. If a line in the image is very long, it must be preserved as a single line in the output, as line breaks (enjambment) are a poetic device.
 - In case of columnar poems, maintain the column structure using spaces in each row to preserve visual structure. Make sure the rows are aligned correctly across all columns.
 - If text is centered or right-aligned, replicate the alignment using spaces so it visually matches the image.
 - If there are gaps within a line (e.g., scattered words or concrete poetry effects), preserve the spacing exactly as in the image.
 - Alignment/indentation: Align word positions precisely with reference lines above/below, preserving exact indentation levels between successive lines. For instance, if the word "foo" in the second line is spaced in a way that the 'f' aligned with the 'b' in the word "bar" in the previous line in the image, then it should be reflected similarly in the text. 
 - In case of newlines/vertical spacing, preserve the exact number of newlines and vertical gaps as seen in the image.
 - In case of concrete poems / scattered poems, the visual layout of the image is a part of the semantics of the poem. Capture it faithfully as possible with spaces.
 - Accurately represent all non-English and special characters (é, ç, ß, etc.) using their exact Unicode code points. Do not use approximations. (e.g., don't replace é with e).
- Use appropriate single Unicode characters for superscripts/subscripts (e.g., ², ₁). 
 - For erasure/blackout poetry, transcribe only the visible text and use spaces to represent the blacked-out areas, preserving the position of the remaining words.
 - In case of page numbers and sections breaks, preserve the layout and spacing exactly as it appears in the image.
 - For superscript/subscript/interpolation of multiple characters, use the appropriate Unicode characters (e.g., ² for superscript 2, ₁ for subscript 1) and ensure they are placed correctly in relation to the surrounding text.
 - In case of rotated/upside-down characters, use the corresponding Unicode character wherever possible.
 - **Ligatures:** Decompose typographic ligatures into their constituent characters (e.g., transcribe 'ﬁ' as 'fi', 'ﬂ' as 'fl', and 'æ' as 'ae').

## Prioritization in Cases of Conflict
All guidelines serve the primary objective, but if rules appear to conflict, follow this strict priority order:
 - **Most Important** Global Layout > Local Spacing: Prioritize the overall "shape" and structure. If maintaining the exact space count between two words causes a column or a centered block to become misaligned, always prioritize the global alignment (the column's starting position, the text's center point) over the exact local space count.
 - **Specific Poem Types > General Rules:** Rules for specific types (like `erasure poetry`) **always override** general formatting rules (like `ignore all... strikethrough`).
 - Visual Alignment > Semantic Characters: The highest priority is to make the text output *look* like the image. Instructions to use specific Unicode characters (like `²` or `₁`) or to decompose ligatures (like `ﬁ` to `fi`) must **be ignored** if following them would alter the character count or width in a way that breaks the poem's visual alignment. In such a conflict, transcribe the characters *exactly as needed to hold the visual shape*, even if it means using standard characters (like `f` and `i` separately) to match the layout.

## Output Format:
- Output must consist of exactly one fenced code block containing only the transcription. Do not include explanations, labels, or commentary outside the block.
- Output must be valid UTF-8 text using only ASCII spaces (U+0020) and standard line breaks (LF: U+000A) for whitespace.
"""


def clean_response_text(text: str) -> str:
    """Removes the markdown code block fences from the API response."""
    lines = (text or "").strip().split('\n')
    if not lines:
        return ""
    if lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return '\n'.join(lines)

def get_image_files(input_dir: Path) -> List[Path]:
    """Get all valid image files from the input directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    return sorted([
        f for f in input_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

def is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection of a rate limit (HTTP 429) error."""
    name = exc.__class__.__name__.lower()
    if "ratelimit" in name or "rate_limit" in name:
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "too many requests" in msg

# --- abstract base class for model handlers ---

class ModelHandler(ABC):
    """Abstract base class for different LLM model handlers."""
    
    def __init__(self, delay_between_calls: float = 1.5):
        self.delay_between_calls = delay_between_calls
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the model client and check API credentials."""
        pass
        
    @abstractmethod
    def process_image(self, image_path: Path) -> str:
        """Process a single image and return the transcribed text."""
        pass
    
    def get_model_name(self) -> str:
        """Get a human-readable model name."""
        return self.__class__.__name__.replace("Handler", "")

# --- model-specific handlers ---

class GeminiHandler(ModelHandler):
    """Handler for Google Gemini models."""
    
    def __init__(self, delay_between_calls: float = 1.5):
        super().__init__(delay_between_calls)
        self.model = None
        
    def initialize(self) -> None:
        if not GEMINI_AVAILABLE:
            raise ImportError("Gemini dependencies not available. Install with: pip install google-generativeai pillow")
        try:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self.model = genai.GenerativeModel('models/gemini-2.5-pro-preview-06-05')
        except KeyError:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
    
    def process_image(self, image_path: Path) -> str:
        if not self.model:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        image = Image.open(image_path)
        response = self.model.generate_content([SYSTEM_PROMPT, image])
        return clean_response_text(response.text)

class ClaudeHandler(ModelHandler):
    """Handler for Anthropic Claude models."""

    def __init__(self, delay_between_calls: float = 2.0):
        super().__init__(delay_between_calls)
        self.client = None
        
    def initialize(self) -> None:
        if not CLAUDE_AVAILABLE:
            raise ImportError("Claude dependencies not available. Install with: pip install anthropic")
        try:
            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        except KeyError:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
    
    def _image_to_base64(self, image_path: Path) -> Tuple[str, str]:
        """Convert image to base64 and determine media type."""
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        extension = image_path.suffix.lower()
        media_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg', 
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(extension, 'image/jpeg')
        return image_data, media_type
    
    def process_image(self, image_path: Path) -> str:
        if not self.client:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        image_base64, media_type = self._image_to_base64(image_path)
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            messages=[{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": SYSTEM_PROMPT
                }, {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_base64
                    }
                }]
            }]
        )
        return clean_response_text(response.content[0].text)

class OpenAIHandler(ModelHandler):
    """Handler for OpenAI models."""
    def __init__(self, delay_between_calls: float = 1.5, max_retries: int = 5):
        super().__init__(delay_between_calls)
        self.client = None
        self.max_retries = max_retries
        self.backoff_initial = 1.0
        self.backoff_max = 30.0
        self.backoff_jitter = 0.25
        
    def initialize(self) -> None:
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI dependencies not available. Install with: pip install openai")
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI()
    
    def _image_path_to_data_url(self, path: Path) -> str:
        """Convert image to data URL format."""
        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            suffix = path.suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
            elif suffix == ".png":
                mime = "image/png"
            elif suffix == ".webp":
                mime = "image/webp"
            else:
                mime = "application/octet-stream"
        with path.open("rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    
    def process_image(self, image_path: Path) -> str:
        if not self.client:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        data_url = self._image_path_to_data_url(image_path)
        # retry loop with exponential backoff for rate limit errors
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model="o3-2025-04-16",
                    reasoning_effort="high",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": SYSTEM_PROMPT.strip()},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }],
                )
                break
            except Exception as api_exc:
                if is_rate_limit_error(api_exc) and attempt < self.max_retries:
                    backoff = min(self.backoff_max, self.backoff_initial * (2 ** (attempt - 1)))
                    backoff += random.uniform(0, self.backoff_jitter)
                    print(f"🔁 Rate limit on {image_path.name}; retry {attempt}/{self.max_retries - 1} in {backoff:.2f}s")
                    time.sleep(backoff)
                    continue
                raise
        else:
            raise RuntimeError("Failed to obtain response after retries")
        raw_text = response.choices[0].message.content or ""
        return clean_response_text(raw_text)

# --- main processing logic ---

def process_images(model_handler: ModelHandler, input_dir: Path, output_dir: Path) -> None:
    """
    Process all images in the input directory using the specified model handler.
    """
    print(f"Starting poem processing with {model_handler.get_model_name()}...")
    # create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_dir.resolve()}")
    # get image files
    image_files = get_image_files(input_dir)
    if not image_files:
        print(f"No images found in '{input_dir}'. Please add your image files and run again.")
        return
    print(f"Found {len(image_files)} images to process.")
    
    # initialize the model
    try:
        model_handler.initialize()
    except (ImportError, ValueError) as e:
        print(f"Error initializing {model_handler.get_model_name()}: {e}")
        return
    
    # process each image
    for i, image_path in enumerate(image_files, start=1):
        print("-" * 40)
        output_filename = image_path.stem + ".txt"
        output_path = output_dir / output_filename
        # skip if already processed
        if output_path.exists():
            print(f"Skipping [{i}/{len(image_files)}] {image_path.name} (already processed).")
            continue
        print(f"Processing [{i}/{len(image_files)}] {image_path.name}...")
        try:
            formatted_text = model_handler.process_image(image_path)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(formatted_text)
            print(f"Success! Saved to {output_path.name}")
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            print("    Moving to the next image.")
        # rate limiting
        time.sleep(model_handler.delay_between_calls)
    
    print("-" * 40)
    print("All images have been processed!")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM OCR for poem image linearization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llm_ocr.py --model gemini
  python llm_ocr.py --model claude --input ../images --output ./results
  python llm_ocr.py --model openai --delay 2.0
        """.strip()
    )
    
    parser.add_argument(
        "--model", 
        choices=["gemini", "claude", "openai"], 
        required=True,
        help="LLM model to use for transcription"
    )
    
    parser.add_argument(
        "--input", 
        type=Path, 
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory containing images (default: {DEFAULT_INPUT_DIR})"
    )
    
    parser.add_argument(
        "--output", 
        type=Path, 
        default=None,
        help="Output directory for transcriptions (default: ./poem_texts/{model}/)"
    )
    
    parser.add_argument(
        "--delay", 
        type=float, 
        default=None,
        help="Delay between API calls in seconds (default varies by model)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    # set default output directory if not specified
    if args.output is None:
        args.output = Path(DEFAULT_OUTPUT_DIR) / args.model
    # create model handler
    handlers = {
        "gemini": GeminiHandler,
        "claude": ClaudeHandler, 
        "openai": OpenAIHandler
    }
    handler_class = handlers[args.model]
    
    # initialize with custom delay if specified
    if args.delay is not None:
        handler = handler_class(delay_between_calls=args.delay)
    else:
        handler = handler_class()
    # process images
    try:
        process_images(handler, args.input, args.output)
    except KeyboardInterrupt:
        print("\n Processing interrupted by user.")
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Path error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()