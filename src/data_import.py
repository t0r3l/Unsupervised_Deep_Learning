"""Import 3 distinct Quick, Draw! classes via the `quickdraw` package.

`quickdraw` exposes three classes (QuickDrawData, QuickDrawDataGroup,
QuickDrawing). We use QuickDrawDataGroup, which downloads a category's
`<name>.bin` from Google Cloud Storage on first use and caches it under
`cache_dir` — so no manual `gsutil` download is needed.

Importable helpers (used by `exploration.ipynb`):
    - load_group(name, max_drawings): download/load one class
    - drawing_to_record(drawing):     flatten a drawing's metadata to a dict
"""

from pathlib import Path

from quickdraw import QuickDrawDataGroup

ROOT = Path(__file__).resolve().parents[1]

# Three distinct classes (categories) to download and inspect.
CLASSES = ["cat", "apple", "car"]
CACHE_DIR = ROOT / "data" / "binary"

HEAD = 3  # number of drawings to load & print per class in the CLI run


def load_group(name: str, max_drawings: int = HEAD) -> QuickDrawDataGroup:
    """Download (on first use) and load one Quick, Draw! class."""
    return QuickDrawDataGroup(
        name, max_drawings=max_drawings, cache_dir=str(CACHE_DIR)
    )


def drawing_to_record(drawing) -> dict:
    """Flatten a QuickDrawing's metadata into a plain dict (nice for DataFrames)."""
    return {
        "key_id": drawing.key_id,
        "country": drawing.countrycode,
        "recognized": drawing.recognized,
        "timestamp": drawing.timestamp,
        "strokes": drawing.no_of_strokes,
    }


def main() -> None:
    for name in CLASSES:
        group = load_group(name)
        print(f"=== {name} (head of {HEAD}) ===")
        for d in group.drawings:
            r = drawing_to_record(d)
            print(
                f"key_id={r['key_id']} country={r['country']} "
                f"recognized={r['recognized']} timestamp={r['timestamp']} "
                f"strokes={r['strokes']}"
            )
        print()


if __name__ == "__main__":
    main()
