"""Dataset preparation independent from TensorFlow and training side effects."""
from pathlib import Path

LABELS = {"Autistic", "Non_Autistic"}

def label_for_filename(filename):
    label = Path(filename).name.split(".", 1)[0]
    if label not in LABELS:
        raise ValueError(f"Unrecognized dataset label: {filename}")
    return label

def labeled_images(directory):
    folder = Path(directory)
    rows = [(p.name, label_for_filename(p.name)) for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not rows:
        raise ValueError(f"No labeled images found in {folder}")
    return rows
