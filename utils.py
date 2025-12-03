"""
Utility functions for ImageNet label mapping and dataset handling.

This module provides functions to map between different ImageNet label formats:
- Mini-ImageNet labels (dataset-specific indices)
- ImageNet-1K labels (canonical 0-999 indices)
- WordNet IDs (e.g., 'n01440764')
- Human-readable class names
"""

import json
import urllib.request


# URLs for ImageNet metadata
IMAGENET_SIMPLE_LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
IMAGENET_CLASS_INDEX_URLS = [
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json",
]


def download_json(url):
    """Download and parse JSON from a URL.
    
    Args:
        url: URL to download JSON from
        
    Returns:
        Parsed JSON data
    """
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode())


def normalize_wordnet_id(wnid):
    """Normalize WordNet ID to standard format (e.g., 'n01440764').
    
    Args:
        wnid: WordNet ID string (may have variations like '0n01440764', 'N01440764')
        
    Returns:
        Normalized WordNet ID in lowercase 'n' prefix format
    """
    wnid = wnid.strip().lower()
    
    # Fix common typo: '0n' -> 'n'
    if wnid.startswith('0n'):
        wnid = wnid[1:]
    
    # Ensure 'n' prefix
    if wnid and not wnid.startswith('n'):
        wnid = f"n{wnid}"
    
    return wnid


def load_imagenet_labels():
    """Load human-readable ImageNet class labels.
    
    Returns:
        List of 1000 human-readable class names (indices 0-999)
    """
    try:
        return download_json(IMAGENET_SIMPLE_LABELS_URL)
    except Exception as exc:
        print(f"Could not load ImageNet labels ({exc}), using indices instead")
        return [f"class_{i}" for i in range(1000)]


def load_wordnet_to_index_mapping():
    """Load mapping from WordNet IDs to ImageNet-1K class indices.
    
    Returns:
        Dictionary mapping WordNet ID (str) to class index (int)
        Example: {'n01440764': 0, 'n01443537': 1, ...}
    """
    last_exc = None
    
    for url in IMAGENET_CLASS_INDEX_URLS:
        try:
            index_data = download_json(url)
            # Convert to {wordnet_id: index} format
            mapping = {normalize_wordnet_id(v[0]): int(k) for k, v in index_data.items()}
            return mapping
        except Exception as exc:
            last_exc = exc
            print(f"Warning: failed to download class index from {url} ({exc})")
    
    raise RuntimeError(f"Failed to download ImageNet class index: {last_exc}") from last_exc


def create_mini_to_imagenet_mapping(dataset):
    """Create mapping from Mini-ImageNet labels to ImageNet-1K indices.
    
    Args:
        dataset: HuggingFace dataset with 'label' feature containing WordNet IDs
        
    Returns:
        Dictionary mapping Mini-ImageNet label (int) to ImageNet-1K index (int)
        Example: {0: 123, 1: 456, 2: 789, ...}
    """
    # Load WordNet ID to ImageNet index mapping
    wnid_to_idx = load_wordnet_to_index_mapping()
    
    # Get label names from dataset
    label_feature = dataset.features.get('label') if hasattr(dataset, 'features') else None
    label_names = getattr(label_feature, 'names', None)
    
    if not label_names:
        # Fallback: assume dataset already uses ImageNet indices
        unique_labels = sorted(set(dataset['label']))
        if unique_labels and max(unique_labels) >= 1000:
            print("Dataset labels already span ImageNet space; using identity mapping.")
            return {label: label for label in unique_labels}
        
        raise RuntimeError(
            "Dataset label metadata missing, cannot align labels to ImageNet indices."
        )
    
    # Build mapping: Mini-ImageNet label index -> ImageNet-1K index
    mapping = {}
    for mini_idx, wnid in enumerate(label_names):
        normalized_wnid = normalize_wordnet_id(wnid)
        
        if normalized_wnid not in wnid_to_idx:
            raise KeyError(f"WordNet ID '{wnid}' not found in ImageNet-1K mapping.")
        
        mapping[mini_idx] = wnid_to_idx[normalized_wnid]
    
    return mapping


def create_imagenet_to_mini_mapping(dataset):
    """Create reverse mapping from ImageNet-1K indices to Mini-ImageNet labels.
    
    Args:
        dataset: HuggingFace dataset with 'label' feature containing WordNet IDs
        
    Returns:
        Dictionary mapping ImageNet-1K index (int) to Mini-ImageNet label (int)
        Example: {123: 0, 456: 1, 789: 2, ...}
    """
    mini_to_imagenet = create_mini_to_imagenet_mapping(dataset)
    # Reverse the mapping
    return {v: k for k, v in mini_to_imagenet.items()}


def get_label_name(label_idx, imagenet_labels=None):
    """Get human-readable name for an ImageNet class index.
    
    Args:
        label_idx: ImageNet-1K class index (0-999)
        imagenet_labels: Optional pre-loaded list of labels (for efficiency)
        
    Returns:
        Human-readable class name
    """
    if imagenet_labels is None:
        imagenet_labels = load_imagenet_labels()
    
    if 0 <= label_idx < len(imagenet_labels):
        return imagenet_labels[label_idx]
    else:
        return f"class_{label_idx}"
