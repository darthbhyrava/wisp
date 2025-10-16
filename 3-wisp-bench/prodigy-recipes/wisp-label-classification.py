import prodigy

from prodigy.components.loaders import Images
from prodigy.util import split_string
from typing import List, Optional


@prodigy.recipe(
    "wisp.classify",
    dataset=("The dataset to use:", "positional", None, str),
    source=("Path to poetry images directory:", "positional", None, str),
    exclude=("Names of datasets to exclude:", "option", "e", split_string),
    remove_base64=("Remove base64 before saving to DB", "flag", None, bool),
)
def wisp_classify(
    dataset: str,
    source: str,
    exclude: Optional[List[str]] = None,
    remove_base64: bool = False,
):
    """
    Multi-label classification for WISP typology whitespace detection in poetry images.
    """
    # load images from directory
    stream = Images(source)
    
    def before_db(examples):
        for eg in examples:
            img = eg.get("image")
            if isinstance(img, str) and img.startswith("data:"):
                eg["image"] = eg.get("path") or None
        return examples

    def add_wisp_options(stream):
        """Add WISP classification options to each image task"""
        for task in stream:
            task["options"] = [
                {"id": "LINE_BREAKS", "text": "Line Breaks: Sample contains newline characters (\\n) delineating typographic lines."},
                {"id": "PREFIX", "text": "Prefix: Sample contains lines which begin with whitespace indentation."},
                {"id": "INTERNAL", "text": "Internal: Sample contains non-standard whitespace within lines, appearing between or within words."},
                {"id": "VERTICAL", "text": "Vertical: Sample has extra vertical spacing (2+ \\n) between lines."},
                {"id": "CONCRETE", "text": "Concrete: The whitespace in the sample contributes to or is part of the meaning of the sample."}
            ]
            # set up for multiple selection
            task["label"] = "Select all WISP elements present in this poem:"
            yield task
    return {
        "dataset": dataset,
        "stream": add_wisp_options(stream),
        "view_id": "choice",  # choice interface for multi-label classification
        "exclude": exclude,
        "before_db": before_db if remove_base64 else None,
        "config": {
            "choice_style": "multiple",  # allow multi choice / multi-label classification
            "choice_auto_accept": False,  # avoid auto-submit, let annotator review selections
            "global_css": ".prodigy-container { max-width: 95vw !important; } .prodigy-card { max-width: 100% !important; } .prodigy-choice { max-width: 100% !important; }"
        }
    }
