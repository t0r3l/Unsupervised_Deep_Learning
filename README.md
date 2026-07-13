# Unsupervised Deep Learning — Quick, Draw!

A school project exploring **unsupervised deep learning** on the
[Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) sketch dataset.
Each drawing is a **28×28 grayscale bitmap on a uniform background**, so all variance
comes from the sketch itself — ideal for clustering. The goal is to learn a
representation (e.g. an autoencoder) and group a handful of categories **without using
the labels**, then evaluate the clusters against the true labels (ARI / NMI).

## Project structure

```
Unsupervised_Deep_Learning/
├── data/                         # datasets (git-ignored, see "Data setup")
│   ├── numpy_bitmap/             # 28x28 .npy bitmaps used by the clustering pipeline
│   └── binary/                   # per-class .bin files (auto-downloaded by src/data_import.py)
├── documentation/
│   └── quickdraw-dataset-master/ # Quick, Draw! docs: categories.txt, parsers, examples
├── src/                          # project source code
│   ├── data_import.py            # download 3 classes via quickdraw & print their head
│   └── exploration.ipynb         # notebook: prints the head (and a sample sketch) per class
├── myenv/                        # local virtual environment (git-ignored)
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

> Both the virtual environment (`myenv/`) and the downloaded datasets
> (`data/binary/`, `data/numpy_bitmap/`, …) are **git-ignored** — they stay local
> and never get committed.

## Prerequisites

- **Python 3.10+** and `pip`
- ~2 GB free disk per handful of categories you download (each `.npy` bitmap file is tens of MB)

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/t0r3l/Unsupervised_Deep_Learning.git
cd Unsupervised_Deep_Learning

# 2. Create and activate a virtual environment (git-ignored)
python3 -m venv myenv
source myenv/bin/activate          # Windows: myenv\Scripts\activate

# 3. Install dependencies (TensorFlow bundles Keras 3)
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify the install
python -c "import tensorflow as tf, keras; print('TF', tf.__version__, '| Keras', keras.__version__)"
```

## Data setup

The folder `documentation/quickdraw-dataset-master/` (unzipped from `quickdraw-dataset-master.zip`)
is only the **documentation repo** — it contains `categories.txt` (the 345 category
names) and example parsers, **not the images themselves**.

The actual images live on Google Cloud Storage as ready-to-use **28×28 numpy bitmaps**.
Download only the 3–6 categories you want for the project, for example:

```bash
mkdir -p data/numpy_bitmap

# Option A — with gsutil (Google Cloud SDK):
gsutil -m cp \
  "gs://quickdraw_dataset/full/numpy_bitmap/cat.npy" \
  "gs://quickdraw_dataset/full/numpy_bitmap/apple.npy" \
  "gs://quickdraw_dataset/full/numpy_bitmap/car.npy" \
  "gs://quickdraw_dataset/full/numpy_bitmap/tree.npy" \
  data/numpy_bitmap/

# Option B — plain HTTPS (URL-encode spaces in multi-word names as %20):
cd data/numpy_bitmap
curl -O "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/cat.npy"
curl -O "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/apple.npy"
cd ../..
```

Each file loads directly as an array of shape `(N, 784)` — reshape to `(N, 28, 28)`:

```python
import numpy as np
cats = np.load("data/numpy_bitmap/cat.npy").reshape(-1, 28, 28)
print(cats.shape)   # e.g. (123202, 28, 28)
```

> **Note:** everything under `data/` is git-ignored, so the datasets stay off the
> repository. Only `.gitkeep` is committed to preserve the folder.

### Import & explore the data

`src/data_import.py` imports **3 distinct classes** (`cat`, `apple`, `car`) via the
[`quickdraw`](https://quickdraw.readthedocs.io) package and prints the **head**
(first few drawings) of each. No manual download step is needed: `QuickDrawDataGroup`
fetches each category's `.bin` from Google Cloud Storage on first use and caches it
under `data/binary/`.

```bash
python src/data_import.py   # downloads (once) then prints key_id / country / recognized / timestamp / #strokes per class
```

Example output:

```
=== cat (head of 3) ===
key_id=5201136883597312 country=VE recognized=True timestamp=1488497110 strokes=9
...
```

For an interactive view, open **`src/exploration.ipynb`**. It reuses the helpers in
`data_import.py` to load each class into a `pandas` DataFrame, prints `df.head()` per
class, and displays a sample sketch from every category.

```bash
jupyter notebook src/exploration.ipynb
```

## Workflow

1. Pick **3–6 categories** and load their `.npy` bitmaps into one array `X` with a
   label vector `y` (labels are kept **only** for the final evaluation).
2. Normalize (`X / 255.0`) and flatten or keep as images.
3. Learn a representation **unsupervised** — e.g. a Keras autoencoder — and take the
   encoder's latent vectors.
4. Cluster the latent vectors (`sklearn.cluster.KMeans`).
5. Evaluate against `y` with
   `adjusted_rand_score` / `normalized_mutual_info_score` from `sklearn.metrics`.

## License

The Quick, Draw! dataset is released by Google under CC BY 4.0. See
`documentation/quickdraw-dataset-master/LICENSE`.
