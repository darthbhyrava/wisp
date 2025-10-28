import argparse
import json
import random

from pathlib import Path
from typing import Dict, List


def load_wisp_mapping(file_path: Path):
    with open(file_path, 'r') as f:
        return json.load(f)


def create_shortlist(mapping: Dict[str, List[str]], 
                    line_breaks: int = 15,
                    vertical: int = 15, 
                    prefix: int = 15,
                    internal: int = 15,
                    concrete: int = 15) -> List[str]:
    """
    Create a shortlist ensuring minimum representation from each category.
    
    Args:
        mapping: Dictionary with category names and poem slugs
        line_breaks: Minimum number from LINE_BREAKS category
        vertical: Minimum number from VERTICAL category  
        prefix: Minimum number from PREFIX category
        internal: Minimum number from INTERNAL category
        concrete: Minimum number from CONCRETE category
        
    Returns:
        List of selected poem slugs
    """
    if not mapping:
        print("Warning: The mapping file is empty; returning empty shortlist.")
        return []
    selected = set()
    # requirements mapping
    requirements = {
        'LINE_BREAKS': line_breaks,
        'VERTICAL': vertical,
        'PREFIX': prefix,
        'INTERNAL': internal,
        'CONCRETE': concrete
    }
    # check which categories are actually available in the mapping
    available_categories = set(mapping.keys())
    required_categories = set(requirements.keys())
    missing_categories = required_categories - available_categories
    if missing_categories:
        print(f"Warning: The following categories are missing from the mapping: {sorted(missing_categories)}")
        print(f"Available categories: {sorted(available_categories)}")
        # remove missing categories from requirements
        for cat in missing_categories:
            del requirements[cat]
    # check if we have any categories left to work with
    if not requirements:
        print("Warning: No valid categories found in mapping, returning all available poems")
        all_poems = set()
        for poems in mapping.values():
            all_poems.update(poems)
        return sorted(list(all_poems))
    # track how many we've selected from each category
    category_counts = {cat: 0 for cat in requirements.keys()}
    # convert lists to sets for easier operations (only for available categories)
    category_sets = {cat: set(poems) for cat, poems in mapping.items() if cat in requirements}
    # first pass: select poems that appear in multiple categories to maximize efficiency
    all_poems = set()
    for poems in mapping.values():
        all_poems.update(poems)
    # calculate how many categories each poem appears in
    poem_category_count = {}
    for poem in all_poems:
        count = sum(1 for cat_poems in mapping.values() if poem in cat_poems)
        poem_category_count[poem] = count
    # sort poems by how many categories they appear in (descending)
    poems_by_popularity = sorted(all_poems, key=lambda p: poem_category_count[p], reverse=True)
    # greedily select poems that help fulfill multiple requirements
    for poem in poems_by_popularity:
        if poem in selected:
            continue
        # check which unfulfilled requirements this poem can help with
        helpful_categories = []
        for cat, req_count in requirements.items():
            if (cat in category_sets and 
                poem in category_sets[cat] and 
                category_counts[cat] < req_count):
                helpful_categories.append(cat)
        # if this poem helps with any unfulfilled requirements, select it
        if helpful_categories:
            selected.add(poem)
            for cat in helpful_categories:
                category_counts[cat] += 1
    # second pass: fill any remaining requirements
    for category, required_count in requirements.items():
        if category not in category_sets:
            # skip categories that don't exist in the mapping
            continue
        if category_counts[category] < required_count:
            # find poems in this category that aren't already selected
            available = category_sets[category] - selected
            needed = required_count - category_counts[category]
            # randomly select the needed poems
            to_add = random.sample(list(available), min(needed, len(available)))
            selected.update(to_add)
            category_counts[category] += len(to_add)
    return sorted(list(selected))


def print_statistics(shortlist: List[str], mapping: Dict[str, List[str]]):
    """Print statistics about the generated shortlist."""
    print(f"\nShortlist Statistics:")
    print(f"Total poems selected: {len(shortlist)}")
    print("\nCategory breakdown:")
    for category, poems in mapping.items():
        count = sum(1 for poem in shortlist if poem in poems)
        print(f"  {category}: {count} poems")
    # show which expected categories are missing
    expected_categories = {'LINE_BREAKS', 'VERTICAL', 'PREFIX', 'INTERNAL', 'CONCRETE'}
    available_categories = set(mapping.keys())
    missing_categories = expected_categories - available_categories
    if missing_categories:
        print(f"\nMissing expected categories: {sorted(missing_categories)}")
    print(f"\nFirst 10 poems in shortlist:")
    for i, poem in enumerate(shortlist[:10]):
        print(f"  {i+1}. {poem}")
    if len(shortlist) > 10:
        print(f"  ... and {len(shortlist) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Create a shortlist of poems from wisp_mapping.json"
    )
    parser.add_argument(
        '-l', '--line-breaks', 
        type=int, 
        default=15,
        help='Minimum number of poems from LINE_BREAKS category (default: 15)'
    )
    parser.add_argument(
        '-v', '--vertical',
        type=int, 
        default=15,
        help='Minimum number of poems from VERTICAL category (default: 15)'
    )
    parser.add_argument(
        '-p', '--prefix',
        type=int, 
        default=15,
        help='Minimum number of poems from PREFIX category (default: 15)'
    )
    parser.add_argument(
        '-i', '--internal',
        type=int, 
        default=15,
        help='Minimum number of poems from INTERNAL category (default: 15)'
    )
    parser.add_argument(
        '-c', '--concrete',
        type=int, 
        default=15,
        help='Minimum number of poems from CONCRETE category (default: 15)'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path(__file__).parent / 'wisp_mapping.json',
        help='Path to wisp_mapping.json file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent / 'shortlist.txt',
        help='Output file for the shortlist (default: shortlist.txt)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducible results'
    )
    args = parser.parse_args()
    # set random seed if provided
    if args.seed:
        random.seed(args.seed)
    # load the mapping data
    try:
        mapping = load_wisp_mapping(args.input)
    except FileNotFoundError:
        print(f"Error: Could not find {args.input}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {args.input}: {e}")
        return 1
    # create the shortlist
    shortlist = create_shortlist(
        mapping,
        line_breaks=args.line_breaks,
        vertical=args.vertical,
        prefix=args.prefix,
        internal=args.internal,
        concrete=args.concrete
    )
    # output the results
    # always write to file (default: shortlist.txt)
    with open(args.output, 'w') as f:
        for poem in shortlist:
            f.write(f"{poem}\n")
    print(f"Shortlist written to {args.output}")
    # also print first few to stdout for verification
    print(f"\nFirst 10 poems in shortlist:")
    for i, poem in enumerate(shortlist[:10]):
        print(f"  {i+1}. {poem}")
    if len(shortlist) > 10:
        print(f"  ... and {len(shortlist) - 10} more")
    print_statistics(shortlist, mapping)
    return 0


if __name__ == '__main__':
    exit(main())