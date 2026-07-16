"""Sauvegarde / chargement des cartes de Kohonen entraînées.

Un modèle = les feature vectors (k, 784) + ses métadonnées d'entraînement,
stockés dans un seul fichier .npz sous src/kohonen/models/.

La forme de la grille (rows, cols) fait partie des métadonnées : sans elle, les
poids ne sont qu'une liste de k prototypes et toute la topologie — donc les
visualisations — est perdue. Le registre du K-means n'a pas ce besoin : ses
centroïdes n'ont pas de position.
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
    """Enregistre les feature vectors + métadonnées. Écrase si le nom existe déjà.

    metadata doit contenir "rows" et "cols" : la grille est indissociable des
    poids, les séparer donnerait un modèle qu'on ne saurait plus afficher.
    """
    missing = {"rows", "cols"} - set(metadata)
    if missing:
        raise ValueError(
            f"metadata doit contenir la forme de la grille ; manque : {sorted(missing)}."
        )

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
