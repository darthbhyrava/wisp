# Dataset: 2,857 Whitespace-Preserved Public Domain Poems

This dataset includes 2,857 poems from the [Poetry Foundation](https://www.poetryfoundation.org/), which were processed to preserve poetic whitespace---a crucial but often-overlooked aspect of poetry.

To retain original formatting, the poems were linearized with [`resiliparse`](https://resiliparse.chatnoir.eu/en/stable/), an HTML parser that supports precise whitespace preservation (e.g., line breaks, indentation, em-spaces, non-breaking spaces, and more).

## Inclusion Criteria

Poems were selected based on their **public domain status**. In the U.S., as of 2025, texts published in 1929 or earlier have entered the public domain. We thus used a a combination of factors:
- Author death year (<= 1929)
- Presence of “public domain” language in copyright metadata
- Publication year (<= 1929)

## About the Dataset

Each row in the dataset represents a single poem and includes the following metadata:

- **`poem_url`**  
  The full URL of the poem on the Poetry Foundation website.

- **`poem_slug`**  
  A URL-safe identifier for the poem (e.g., `the-red-wheelbarrow`), typically derived from the poem title.

- **`poem_name`**  
  The full title of the poem, as displayed on the Poetry Foundation website.

- **`poem_type`**  
  A label indicating the genre or format of the work (e.g., `"Prose"`, `"Verse"`, `"Visual"`).

- **`poem_text`**  
  The whitespace-preserved full text of the poem, extracted using [`resiliparse`](https://resiliparse.chatnoir.eu/en/stable/). Includes line breaks, indentation, non-breaking spaces, and other whitespace variants relevant to poetry.

- **`poem_copyright_and_source`**  
  Copyright information and original publication/source metadata. Used to determine public domain status when combined with author death year and copyright year.

- **`poem_categories`**  
  Tags or categories assigned by the Poetry Foundation, which may be subjects or forms (e.g., "nature", "love", "sonnet").

- **`poet_name`**  
  The full name of the poet.

- **`poet_pseudonym`**  
  An alternate name used by the poet, if listed (may be blank).

- **`poet_slug`**  
  A URL-safe version of the poet's name (e.g., `emily-dickinson`).

- **`poet_url`**  
  The full URL of the poet’s profile page on the Poetry Foundation website.

- **`poet_birth_year`**  
  The birth year of the poet, as listed on the Poetry Foundation site.

- **`poet_death_year`**  
  The death year of the poet, as listed on the Poetry Foundation site.

- **`birth_year_from_exp`**  
  An extracted birth year, parsed programmatically from textual metadata.

- **`death_year_from_exp`**  
  An extracted death year, parsed programmatically from textual metadata. Used to help assess public domain status.

- **`poet_categories`**  
  Tags or labels applied to the poet (e.g., "Modern", "England," Victorian", "Renaissance").

- **`poem_audio_id`**  
  An identifier for the audio recording of the poem (if available).

- **`audio_slug`**  
  A URL-safe identifier for the audio file (if available).

- **`audio_info`**  
  Information about the audio recording, such as narrator.

- **`issue_id`**  
  The internal Poetry Foundation issue identifier (if the poem appeared in *Poetry* magazine).

- **`issue_month_year`**  
  The month and year the poem appeared in *Poetry* magazine, if applicable.


We also share a [Google spreadsheet](https://docs.google.com/spreadsheets/d/1mr6J3EJKkhMU-u__WbzTftUzLuTgz9lhu5S4uGL7Lcs/edit?usp=sharing) version of this data.

## Individual Poem Files in `.txt` format

In addition to the CSV dataset, we provide the individual poem files in a formatting-preserved version:

- **`poem_texts_resiliparse.zip`**: Contains 2,857 plain text files (`.txt`), one per poem, with whitespace and formatting preserved from original HTML using [`resiliparse`](https://resiliparse.chatnoir.eu/en/stable/). These files attempt to maintain the original line breaks, indentation, and spacing from the Poetry Foundation website (look at WISP-Bench for more details).

These files correspond exactly to the poems listed in `public_domain_shortlist.txt` and provide researchers with access to the full formatting details that are crucial for analyzing poetic structure and whitespace usage.