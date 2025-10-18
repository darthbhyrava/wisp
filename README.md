Code, dataset, and dashboard for the EMNLP paper **"so much depends / upon / a whitespace: Why Whitespace Matters for Poets and LLMs"**  

<p align="center">
    <img src="./assets/cummings-figure.png" alt="Buffalo Bill's, E. E. Cummings" width="350"/>
</p>


## I. Public Domain Poems Dataset

`1-public_domain_poems` contains 2,857 whitespace-preserved public domain poems from the [Poetry Foundation](https://www.poetryfoundation.org/). The poems were selected based on their public domain status (author death year <= 1929) and processed to preserve poetic whitespace using the [`resiliparse`](https://resiliparse.chatnoir.eu/en/stable/) HTML parser.

Please see the [1-public_domain_poems/README.md](./1-public_domain_poems/README.md) for more details about the dataset, including inclusion criteria, metadata fields, and explanation of the individual poem files in `.txt` format.


## II. WISP-Bench: Evaluating Whitespace Preservation Fidelity across Linearization methods.

When you convert a formatted whitespace-rich source text (like a poem) from HTML/Image/PDF/any-other-format to plain text, how can you quantify the quality of whitespace preservation fidelity? We propose WISP-Bench, a benchmark for evaluating whitespace preservation fidelity across different linearization methods.

<p align="center">
    <img src="./assets/2wispbench.png" alt="Prodigy Annotation Task 2" width="750"/>
</p>

WISP-Bench consists of a three-tiered set of pass-or-fail unit-tests that evaluate whitespace preservation across four key dimensions: line breaks, prefix spacing (indentation), internal spacing (between words), and vertical spacing (blank lines). The benchmark includes Prodigy annotation recipes for manual evaluation, tools for generating text from images using multimodal LLM OCR, and comprehensive scoring and visualization scripts. Please see [2-wisp-bench/README.md](./2-wisp-bench/README.md) for detailed setup instructions, annotation guidelines, and usage examples.

## III. WISP Unspacer: Reversible Poem-to-Prose Conversion Tool

Poetic whitespace can confuse language models and linguistic analyzers trained on prose text. The WISP Unspacer is a tool that converts poems to a prose format by reversibly replacing all whitespace characters with single spaces, allowing for accurate linguistic processing. After processing, annotations can be mapped back to the original poem format. Please see [3-wisp-unspacer/README.md](./3-wisp-unspacer/README.md) for installation instructions, usage examples, and chunk structure details.

## IV. Whitespace Patterns in Poetry: [poetry.darthbhyrava.com](https://poetry.darthbhyrava.com)

Explore whitespace patterns in poetry across poems from [The Poetry Foundation](https://www.poetryfoundation.org) in our interactive dashboard at [poetry.darthbhyrava.com](https://poetry.darthbhyrava.com).

<p align="center">
    <img src="./assets/dashboard_snippet.png" alt="Dashboard Screenshot" width="950" />
</p>

The dashboard allows you to visualize and analyze how poets use whitespace as a literary device, offering insights into the spatial dimensions of poetic expression. Browse through different poems, categories of poets, compare poet styles, and discover how formatting contributes to meaning and aesthetic effect. The dashboard only uses and presents features derived from the whitespace patterns of poems, not their textual content, ensuring that we adhere to copyright restrictions.

## V. Citation

This dataset supports the analysis in our paper:  
> “Whitespace.”

---

