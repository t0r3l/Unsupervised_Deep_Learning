"""Sauvegarde / chargement des VAE entraînés.

Un modèle = le vecteur plat de tous ses tenseurs Keras (voir utils/model.py)
+ ses métadonnées — dont l'architecture complète, sans laquelle le vecteur est
indéchiffrable — stockés dans un seul .npz sous src/vae/models/.

Même format que les registres K-means, Kohonen, PCA et autoencodeur : chaque
algo garde le sien, c'est ce qui empêche les modèles de se mélanger entre algos
(un VAE et un autoencodeur simple ont des poids de tailles différentes — deux
têtes latentes contre une).
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


def save_model(name, weights, metadata):
    """Enregistre les poids + métadonnées. Écrase si le nom existe déjà."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights = np.asarray(weights, dtype=np.float32)

    np.savez_compressed(
        model_path(name),
        weights=weights,
        metadata=json.dumps(metadata),
    )
    return model_path(name)


def load_model(name):
    """Retourne (weights, metadata) du modèle nommé."""
    path = model_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")

    data = np.load(path, allow_pickle=False)
    weights = data["weights"]
    metadata = json.loads(data["metadata"].item())
    return weights, metadata


def delete_model(name):
    path = model_path(name)
    if path.exists():
        path.unlink()
