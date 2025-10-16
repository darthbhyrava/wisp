import base64
import json
import os
import prodigy

from pathlib import Path
from prodigy.util import split_string
from typing import List, Optional, Dict, Any

# define sub-labels for each WISP label unit-test
WISP_SUBLABELS = {
    "LINE_BREAKS": [
        {"id": "line_breaks_presence", "text": "Line Breaks Presence: Does the text capture line breaks where they should be?"}
    ],
    "PREFIX": [
        {"id": "prefix_presence", "text": "Prefix Presence: Is indentation preserved at all?"},
        {"id": "prefix_fuzzy", "text": "Prefix Fuzzy: Are relative indentation levels preserved?"},
        {"id": "prefix_exact", "text": "Prefix Exact: Are exact indentation levels preserved?"}
    ],
    "INTERNAL": [
        {"id": "internal_presence", "text": "Internal Presence: Is extra spacing between words preserved?"},
        {"id": "internal_fuzzy", "text": "Internal Fuzzy: Are relative internal spacing levels preserved?"},
        {"id": "internal_exact", "text": "Internal Exact: Are exact internal spacing amounts preserved?"}
    ],
    "VERTICAL": [
        {"id": "vertical_presence", "text": "Vertical Presence: Is vertical spacing (>1 newline) preserved?"},
        {"id": "vertical_fuzzy", "text": "Vertical Fuzzy: Are relative vertical spacing levels preserved?"},
        {"id": "vertical_exact", "text": "Vertical Exact: Are exact vertical spacing amounts preserved?"}
    ]
}

def load_wisp_mapping(mapping_file: str) -> Dict[str, List[str]]:
    # load WISP label to slug mapping
    with open(mapping_file, 'r') as f:
        return json.load(f)

def load_shortlist(shortlist_file: str) -> List[str]:
    # load list of poem slugs to process
    with open(shortlist_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def get_labels_for_slug(slug: str, wisp_mapping: Dict[str, List[str]]) -> List[str]:
    # find which WISP labels this slug belongs to
    labels = []
    for label, slugs in wisp_mapping.items():
        if slug in slugs:
            labels.append(label)
    return labels

def get_combined_sublabels(labels: List[str]) -> List[Dict[str, str]]:
    # combine sublabels from all relevant WISP labels
    combined = []
    for label in labels:
        if label in WISP_SUBLABELS:
            combined.extend(WISP_SUBLABELS[label])
    return combined

def image_to_base64(image_path: str) -> str:
    # convert image to base64 data URI
    with open(image_path, 'rb') as f:
        image_data = f.read()
    encoded = base64.b64encode(image_data).decode()
    return f"data:image/png;base64,{encoded}"

def load_text_file(text_path: str) -> str:
    # load text content from file
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"[Text file not found: {text_path}]"

@prodigy.recipe(
    "wisp.sublabel",
    dataset=("Dataset to save annotations", "positional", None, str),
    mapping_file=("Path to WISP mapping JSON file", "positional", None, str),
    shortlist_file=("Path to shortlist.txt file", "positional", None, str),
    images_dir=("Directory containing poem images", "positional", None, str),
    texts_dir=("Directory containing poem texts", "positional", None, str),
    cat_dir=("Category subdirectory in texts_dir", "positional", None, str),
    exclude=("Datasets to exclude", "option", "e", split_string),
)
def wisp_sublabel_annotation(
    dataset: str,
    mapping_file: str,
    shortlist_file: str,
    images_dir: str,
    texts_dir: str,
    cat_dir: str,
    exclude: Optional[List[str]] = None,
):
    """
    A custom prodigy recipe for WISP sub-label annotation comparing images to linearized text.
    Shows image and text side-by-side with sub-label options based on detected WISP elements.
    Uses shortlist.txt to determine which poems to process and automatically finds relevant WISP labels.
    """
    # load WISP mapping and shortlist
    wisp_mapping = load_wisp_mapping(mapping_file)
    shortlist_slugs = load_shortlist(shortlist_file)
    def create_stream():
        """Generate annotation tasks"""
        for slug in shortlist_slugs:
            # find which WISP labels this slug belongs to
            labels = get_labels_for_slug(slug, wisp_mapping)
            if not labels:
                print(f"Warning: No WISP labels found for slug {slug}")
                continue
            # get combined sublabels for all relevant WISP labels
            combined_sublabels = get_combined_sublabels(labels)
            if not combined_sublabels:
                print(f"Warning: No sublabels found for slug {slug} with labels {labels}")
                continue
            # construct file paths
            image_path = os.path.join(images_dir, f"{slug}.png")
            text_path = os.path.join(texts_dir, cat_dir, f"{slug}_poem_text.txt")
            # check if files exist
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            if not os.path.exists(text_path):
                print(f"Warning: Text file not found: {text_path}")
                continue
            # load image and text
            image_b64 = image_to_base64(image_path)
            text_content = load_text_file(text_path)

            # create task (provide raw data only; template handles layout)
            task = {
                "image_b64": image_b64,  # transient – removed before DB
                "poem_text": text_content,  # transient – removed before DB
                "wisp_labels": labels,  # transient – removed before DB
                "options": combined_sublabels,
                # 'text' is required by the choice view even if we render custom HTML.
                # Use the filename (slug) plus labels for hashing / display consistency.
                # Use slug as text so tasks get unique hashes (previous constant caused deduplication)
                "text": slug,
                "filename": slug,
                # Store original image path for persistence / review if base64 removed later
                "image": image_path,
                "meta": {
                    "image_path": image_path,
                    "text_path": text_path,
                    "wisp_labels": labels,
                    "category": cat_dir
                }
            }
            yield task

    def before_db(examples):
        """Strip transient large fields to minimize DB size."""
        for eg in examples:
            for k in ("image_b64", "poem_text", "wisp_labels"):
                if k in eg:
                    del eg[k]
        return examples
    
    return {
        "dataset": dataset,
        "stream": create_stream(),
                "view_id": "blocks",
        "exclude": exclude,
        "before_db": before_db,
        "config": {
            "choice_style": "multiple",
            "choice_auto_accept": False,
            "global_css": ".prodigy-container { max-width: 100% !important; }",
            "blocks": [
                {"view_id": "html", "html_template": """
                <div style='display:flex; gap:24px; margin-bottom:20px; width:100%; align-items:flex-start;'>
                    <!-- 50% image column -->
                    <div id='image-container' style='flex:1 1 50%; max-width:50%;'>
                        <h3 style='margin-top:0;'>Image</h3>
                        <div style='width:100%; border:1px solid #ccc; background:white; display:flex; justify-content:center;'><img src='{{image_b64}}' style='width:100%; height:auto; object-fit:contain;' onload='adjustTextHeight()'/></div>
                    </div>
                    <!-- 50% text column -->
                    <div style='flex:1 1 50%; max-width:50%; min-width:0;'>
                        <h3 style='margin-top:0;'>Text ({{meta.category}})</h3>
                    <pre id='text-container' style='border:1px solid #ccc; padding:10px; margin:0; background:white; color:black; font-family:monospace; white-space:pre; overflow-x:auto; overflow-y:auto; box-sizing:border-box; width:100%; text-align:left;'>{{poem_text}}</pre>
                    </div>
                </div>
                <script>
                function adjustTextHeight(){
                    const img=document.querySelector('#image-container img');
                    const txt=document.getElementById('text-container');
                    if(img && txt){ txt.style.height = img.offsetHeight + 'px'; }
                }
                window.addEventListener('resize', adjustTextHeight);
                </script>
                """},
                {"view_id": "choice"}
            ]
        }
    }