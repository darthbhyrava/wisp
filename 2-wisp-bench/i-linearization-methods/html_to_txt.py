from bs4 import BeautifulSoup
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import re
from bs4 import NavigableString
import unicodedata
import html

# Define a regex pattern for all Unicode space variants
UNICODE_WHITESPACE_PATTERN = re.compile(
    r'[\u00A0\u1680\u180E\u2000-\u200A\u202F\u205F\u3000]'
)

def normalize_unicode_spaces(text, replacement='SPACE'):
    """Replaces all uncommon Unicode space characters with a placeholder or normal space."""
    return UNICODE_WHITESPACE_PATTERN.sub(replacement, text)

# Custom translation for known ligatures
LIGATURE_MAP = {
    '\ufb00': 'ff',
    '\ufb01': 'fi',
    '\ufb02': 'fl',
    '\ufb03': 'ffi',
    '\ufb04': 'ffl',
    '\ufb05': 'ft',
    '\ufb06': 'st',
}

def normalize_ligatures(text):
    return text.translate(str.maketrans(LIGATURE_MAP))

# Convert margin-left into actual spaces
def extract_margin_left(style_str):
    """Extract margin-left in px from a style string, return as number of spaces."""
    match = re.search(r'margin-left:\s*(\d+)px', style_str)
    if match:
        px = int(match.group(1))
        # Rough rule: 1 space per 10px 
        return 'WHITE' * (px // 10)
    return ''


def convert_html_to_text(html_content):
    """
    Convert HTML to text, preserving poetic formatting.
    Handles:
    - Centered poems via <center>
    - Line breaks via <br> and <div>
    - Fallback when .poem-body is missing
    """

    # for handling unicode characters like weird spaces and ligatures
    html_content = html.unescape(html_content)

    soup = BeautifulSoup(html_content, 'lxml')
    poem_div = soup.find("div", class_="poem-body")

    # Convert margin-left into actual spaces
    for div in poem_div.find_all('div', style=True):
        indent = extract_margin_left(div['style'])
        if indent:
            div.insert(0, NavigableString(indent))
        del div['style']
    
    # Convert small caps to caps 
    for span in poem_div.find_all('span', class_='sm-caps'):
        span.string = span.get_text().upper()

    # Fallback: no poem-body present
    if not poem_div:
        return normalize_ligatures(soup.get_text())

    # Case 1: Centered poem
    if poem_div.find('center'):
        centered_lines = []
        for center in poem_div.find_all("center"):
            line = center.get_text().strip()
            centered_lines.append(line.center(80))  # visually center
            center.decompose()  # remove tag after extracting text
        return normalize_ligatures("\n".join(centered_lines))

    # Check for divs or ps 
    inner_divs = poem_div.find_all('div', recursive=True)
    inner_paras = poem_div.find_all('p', recursive=False)

    if inner_divs:
        
    # Case 2: Line-broken poem with <br> and nested <div>
        for tag in poem_div.find_all(['i', 'em', 'b']):
            # Check if tag starts a line: nothing or only whitespace/br/div before it
            prev = tag.previous_sibling

            is_line_start = (
                prev is None or
                (isinstance(prev, NavigableString) and prev.strip() == '') or
                (getattr(prev, 'name', None) in ['br', 'div'])
            )

            if is_line_start:
                #tag.insert_after('WHITE')
                continue
            else:
                tag.insert_before('WHITE')
                tag.insert_after('WHITE')


        # Add BREAK markers around divs for line breaks
        for div in poem_div.find_all('div'):
            # Rule: if the div contains only whitespace or &nbsp;, plus <br>, treat as a single BREAK
            contents = list(div.contents)
            if (
                all(
                    (isinstance(c, NavigableString) and not c.strip().replace('\xa0', '')) or
                    (getattr(c, 'name', None) == 'br')
                    for c in contents
                )
            ):
                div.clear()
                div.append('BREAK')
                continue

            # Insert a BREAK before every <div> to denote potential line break
            div.insert_before('BREAK')

            # Check if the <div> has direct (non-nested) text content
            direct_text = div.find(text=True, recursive=False)

            # Check if this <div> contains inner <div>s (i.e., it's a parent with nested structure)
            inner_divs = div.find_all('div', recursive=False)

           # Don't insert a BREAK before direct text if there's only one inner div and it's the first child
            if (
                direct_text
                and direct_text.strip('\xa0 ').strip()  # contains visible content
                and inner_divs
            ):
                # Only insert a BREAK if the inner div is not at the start or there's multiple
                i = div.contents.index(direct_text)
                divs_before = [c for c in div.contents[:i] if getattr(c, 'name', None) == 'div']
                if not (len(inner_divs) == 1 and divs_before == [inner_divs[0]]):
                    direct_text.insert_before('')

            # If the div contains inner divs and its last element is a <br>, insert BREAK — but only if that <br> isn’t already being handled
            if inner_divs:
                last = div.contents[-1] if div.contents else None
                has_non_div_text = any(
                    isinstance(c, NavigableString) and c.strip('\xa0 ').strip()
                    for c in div.contents
                    if not getattr(c, 'name', None) == 'div'
                )
                if last and getattr(last, 'name', None) == 'br' and not has_non_div_text:
                    div.append('BREAK')

            # If the next sibling after the <div> is a text node and not just whitespace,
            # insert a BREAK after this <div> as well, to preserve visual separation
            next_sibling = div.next_sibling
            if isinstance(next_sibling, NavigableString) and next_sibling.strip():
                div.insert_after('BREAK')

        # Remove <br> tags
        for br in poem_div.find_all('br'):
            br.replace_with('')

        # Unwrap all <div> tags so their contents become part of the flat text structure
        for div in poem_div.find_all('div'):
            div.unwrap()


        # Collect all text nodes in the poem (now unwrapped)
        text_nodes = [el for el in poem_div.descendants if isinstance(el, NavigableString)]

        for element in text_nodes:
            raw_text = str(element)
            # Remove stray whitespace characters for clean processing
            stripped = raw_text.strip('\n\r\t ')

            if stripped == '\xa0':
                # If the node is just a non-breaking space (&nbsp;), treat it as a line break
                element.replace_with('BREAK')
                #continue
            else:
                # Otherwise, replace any remaining &nbsp; in the string with a marker ("SPACE")
                # so we can later turn them back into non-breaking spaces for formatting
                element.replace_with(raw_text.replace('\xa0', 'SPACE'))

       
        # Get final text
        text = poem_div.get_text(strip=True)
        text = normalize_unicode_spaces(text, replacement='SPACE')
        text = (
           text.replace("BREAK", "\n")
                .replace("WHITE", " ")
                .replace("SPACE", "\xa0")
        )

        return normalize_ligatures(re.sub(r'^\n+', '', text))
    
    # ONE PARAGRAPH TAG
    elif len(inner_paras) == 1 and inner_paras[0].find_all('br'):
        # Case: Single paragraph with <br> breaks
        for br in poem_div.find_all('br'):
            br.replace_with('BREAK')
        for tag in poem_div.find_all(['p']):
            tag.unwrap()
        
        for tag in poem_div.find_all(['i', 'em', 'b']):
            # Check if tag starts a line: nothing or only whitespace/br/div before it
            prev = tag.previous_sibling

            is_line_start = (
                prev is None or
                (isinstance(prev, NavigableString) and prev.strip() == '') or
                (getattr(prev, 'name', None) in ['br', 'div'])
            )

            if is_line_start:
                continue
            else:
                tag.insert_before('WHITE')
                tag.insert_after('WHITE')

        text = poem_div.get_text(strip=True)
        text = normalize_unicode_spaces(text, replacement='SPACE')

        text = (
            text.replace("BREAK", "\n")
                .replace("SPACE", "\xa0")
                .replace("WHITE", " ")

        )
        return normalize_ligatures(text)

    # MORE THAN ONE PARAGRAPH
    elif len(inner_paras) > 1:
        for tag in poem_div.find_all(['i', 'em', 'b']):
            # Check if tag starts a line: nothing or only whitespace/br/div before it
            prev = tag.previous_sibling

            is_line_start = (
                prev is None or
                (isinstance(prev, NavigableString) and prev.strip() == '') or
                (getattr(prev, 'name', None) in ['br', 'div'])
            )

            if is_line_start:
                #tag.insert_after('WHITE')
                continue
            else:
                tag.insert_before('WHITE')
                tag.insert_after('WHITE')
        # Case 3: Multiple <p> paragraphs, no inner divs or line breaks
        paragraphs = []
        for p in inner_paras:
            paragraphs.append(p.get_text(strip=True).replace('\xa0', ' '))
        text = "\n".join(paragraphs)
        text = normalize_unicode_spaces(text, replacement='SPACE')
        text = (
           text.replace("BREAK", "\n")
                .replace("WHITE", " ")
                .replace("SPACE", "\xa0")
        )
        return normalize_ligatures(text)

    else:
        # Fallback
        return normalize_ligatures(poem_div.get_text())

    # # Case 3: Just cleanly strip the .poem-body text
    # return normalize_ligatures(poem_div.get_text())

def process_single_file(args):
    input_path, output_path = args
    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        text_content = convert_html_to_text(html_content)
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text_content)
        return f"Processed: {os.path.basename(input_path)}"
    except Exception as e:
        return f"Error processing {os.path.basename(input_path)}: {str(e)}"

def process_html_files(input_dir, output_dir, max_workers=None):
    """
    Process all HTML files in a directory in parallel, save as text files.
    max_workers: number of parallel processes to use. Defaults to 2.
    """
    # create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if max_workers is None:
        # max_workers = multiprocessing.cpu_count()
        max_workers = 2
    # get list of files to process
    file_pairs = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.html'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.html', '.txt'))
            file_pairs.append((input_path, output_path))
    # process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_single_file, file_pairs)
        # print results as they complete
        for result in results:
            print(result)

if __name__ == "__main__":
    input_directory = "output/poem_text/"
    # input_directory = "htmls"
    output_directory = "revised_converted_htmls/"
    # output_directory = "converted_htmls/"
    process_html_files(input_directory, output_directory, max_workers=4)