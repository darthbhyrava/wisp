# WISP Unspacer

If you try to feed the text of a poem into a language model or linguistic analyzer for something as rudimentary as part-of-speech tagging or syntactic parsing, you may find the results disappointing. Since these models are typically trained on prose text with standard spacing conventions, they ignore the unique whitespace patterns found in poetry. Worse still, they often misinterpret the presence of whitespace, using them as markers of sentence boundaries or semantic stops (there goes enjambment!), leading to garbled analyses.

To fix this, we propose a solution: the WISP Unspacer. This converts the poem to a prose format by reversibly replacing all whitespace characters (spaces, tabs, newlines) with single spaces. This allows language models and linguistic analyzers to process the poem as a continuous stream of text (prose). Once we have our linguistic annotations, we can then map them back to the original poem format by reversing the unspacing process.

The models see prose text for their linguistic annotations, but we still have our whitespace-rich poem and can now map annotations across the two formats.

**Note:** A contiguous pair of text unit + following whitespace unit is called a "chunk".


## Basic Usage

(make sure you have regex installed)

```python
from wisp_unspacer import WISPunspacer

# original poem with complex whitespace
poem = """The    old pond
    a frog jumps      in

sound of water."""
# create unspacer instance
unspacer = WISPunspacer(poem)
# get prose version for NLP processing
prose_text = unspacer.prose
print(prose_text)
# Output: "The old pond a frog jumps in sound of water."
# verify reversibility
assert unspacer.reversible == True
assert unspacer.reconstructed_poem == poem
```

### Chunk Structure

Each chunk in `.chunks` contains:

```python
{
    "text": "Word",                   # Non-whitespace content
    "ws": "    ",                     # Following whitespace
    "wisp_labels": ["internal"],      # Whitespace classification in WISP labels
    "newline_count": 0                # Count of '\n' characters
}
```

### Parsing Example

For input `"Hello\n  world!"`, the parsing process:
1. Regex tokens: `["Hello", "\n  ", "world!"]`
2. Becomes chunks:
```python
[
    { "text": "Hello", "ws": "\n  ", "wisp_labels": ["line_break", "prefix"], "newline_count": 1 },
    { "text": "world!", "ws": "", "wisp_labels": [], "newline_count": 0 }
]
```



