import json
import os

from collections import defaultdict
from pathlib import Path

def extract_wisp_mapping(annotations_file, output_dir="."):
    annotations = []
    with open(annotations_file, 'r') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    # WISP_label -> list of poem slugs
    wisp_files = defaultdict(list)
    print(f"Processing {len(annotations)} annotations...")
    for ann in annotations:
        # only process accepted annotations that have selected labels
        if ann.get('answer') == 'accept' and 'accept' in ann:
            # extract poem slug from multiple possible sources
            slug = None
            # try to get slug from different fields
            if 'text' in ann and ann['text']:
                # use the text field
                slug = ann['text']
            elif 'meta' in ann and 'file' in ann['meta']:
                # extract from filename, removing .png extension
                slug = Path(ann['meta']['file']).stem
            elif 'path' in ann:
                # extract from path
                slug = Path(ann['path']).stem
            if not slug:
                print(f"Warning: Could not extract slug from annotation: {ann.get('_input_hash', 'unknown')}")
                continue
            # get selected WISP labels from the 'accept' field
            selected_labels = ann['accept']
            if isinstance(selected_labels, str):
                selected_labels = [selected_labels]
            # filter for valid WISP labels
            valid_labels = set()
            for label in selected_labels:
                if label in ['LINE_BREAKS', 'PREFIX', 'INTERNAL', 'VERTICAL', 'CONCRETE']:
                    valid_labels.add(label)
            # add slug to mapping for each selected label
            for label in valid_labels:
                if slug not in wisp_files[label]:
                    wisp_files[label].append(slug)
    # create output directory and save json
    os.makedirs(output_dir, exist_ok=True)
    mapping_file = os.path.join(output_dir, "wisp_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(dict(wisp_files), f, indent=2)
    print(f"Saved complete mapping to {mapping_file}")
    return dict(wisp_files)

def analyze_mapping(mapping):
    """
    Analyze the WISP mapping for interesting patterns
    """
    print(f"\nMAPPING ANALYSIS:")
    print("=" * 50)
    # samples with multiple WISP elements
    file_to_labels = defaultdict(set)
    for label, files in mapping.items():
        for file in files:
            file_to_labels[file].add(label)
    # statistics
    multi_label_files = {f: labels for f, labels in file_to_labels.items() if len(labels) > 1}
    single_label_files = {f: labels for f, labels in file_to_labels.items() if len(labels) == 1}
    print(f"samples with multiple WISP elements: {len(multi_label_files)}")
    print(f"samples with single WISP element: {len(single_label_files)}")
    # unique samples count
    print(f"\ntotal unique samples: {len(file_to_labels)}")
    # label counts
    print(f"\nlabel counts:")
    label_counts = defaultdict(int)
    for labels in file_to_labels.values():
        for label in labels:
            label_counts[label] += 1
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count} samples")
    # label co-occurrence
    print(f"\nLabel co-occurrence patterns:")
    label_pairs = defaultdict(int)
    for labels in multi_label_files.values():
        labels_list = sorted(labels)
        for i in range(len(labels_list)):
            for j in range(i+1, len(labels_list)):
                pair = (labels_list[i], labels_list[j])
                label_pairs[pair] += 1
    for pair, count in sorted(label_pairs.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {pair[0]} + {pair[1]}: {count} samples")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 wisp_mapping.py <annotations.jsonl>")
        print("Example: python3 wisp_mapping.py wisp_annotations.jsonl")
        sys.exit(1)
    annotations_file = sys.argv[1]
    if not os.path.exists(annotations_file):
        print(f"Error: File '{annotations_file}' not found")
        sys.exit(1)
    mapping = extract_wisp_mapping(annotations_file)
    analyze_mapping(mapping)