"""Sauvegarde / chargement des K-means entraînés.

Un modèle = les centroïdes (k, 784) + ses métadonnées d'entraînement,
stockés dans un seul fichier .npz sous src/kmeas/models/.
"""

import json
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def model_path(name):
    return MODELS_DIR / f"{name}.npz"


def list_models():
    """Noms des modèles disponibles, triés."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.stem for p in MODELS_DIR.glob("*.npz"))


def save_model(name, centroids, metadata):
    """Enregistre les centroïdes + métadonnées. Écrase si le nom existe déjà."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    centroids = np.asarray(centroids, dtype=np.float32)

    np.savez_compressed(
        model_path(name),
        centroids=centroids,
        metadata=json.dumps(metadata),
    )
    return model_path(name)


def load_model(name):
    """Retourne (centroids, metadata) du modèle nommé."""
    path = model_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")

    data = np.load(path, allow_pickle=False)
    centroids = data["centroids"]
    metadata = json.loads(data["metadata"].item())
    return centroids, metadata


def delete_model(name):
    path = model_path(name)
    if path.exists():
        path.unlink()
