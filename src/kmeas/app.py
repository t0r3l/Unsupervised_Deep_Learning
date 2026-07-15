"""App Gradio — explorer le K-means comme codec (quantification vectorielle).

Lancement, depuis src/kmeas/ :
    python app.py

Le dataset se choisit dans l'app (MNIST ou Quick, Draw!) et vaut pour tous les
onglets. Le chargement passe par src/data_import.py, l'entrée unique du projet :
tous les datasets en ressortent au même format (n, 784) float32 dans [0, 1].

Les modèles sauvegardés portent leur algo et leur dataset dans leurs métadonnées
(voir ALGO plus bas) : le sélecteur ne propose que ceux qui correspondent à la
vue courante. C'est ce qui permettra de brancher d'autres algos sans que leurs
modèles se mélangent à ceux du K-means.
"""

import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib

# Backend non-interactif : le serveur rend les figures en PNG sans jamais ouvrir
# de fenêtre. À poser avant d'importer pyplot, sinon matplotlib choisit un
# backend GUI et plante hors du thread principal.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import gradio as gr

# data_import.py vit dans src/, ce fichier dans src/kmeas/ : sans ça, `import
# data_import` échoue selon le dossier depuis lequel on lance l'app.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_import import DATASETS, load_dataset  # noqa: E402

from utils.codec import decode, encode  # noqa: E402
from utils.kmeas import (  # noqa: E402
    assign_clusters,
    compute_inertia,
    compute_squared_distances,
    fit_kmeans,
)
from utils.metrics import compute_compression_ratio, mse_from_inertia  # noqa: E402
from utils.registry import delete_model, list_models, load_model, save_model  # noqa: E402
from utils.visualization import (  # noqa: E402
    plot_centroid_map,
    plot_class_distribution,
    plot_inertia,
    plot_latent_space,
    show_images,
    show_reconstructions,
)

IMAGE_DIM = 784
DEFAULT_DATASET = "mnist"

# Algo servi par cette app. Estampillé dans les métadonnées de chaque modèle pour
# qu'un futur autoencodeur, dont les poids n'ont rien à voir avec des centroïdes,
# ne se retrouve jamais proposé dans le sélecteur du K-means.
ALGO = "kmeans"

# Les datasets restent en mémoire une fois chargés : MNIST coûte ~10 s (import
# TensorFlow), et on ne modifie jamais ces tableaux.
_LOADED = {}


def get_dataset(key, progress=None):
    """Charge un dataset (ou le ressort du cache). Entrée unique de l'app."""
    if key not in _LOADED:
        _LOADED[key] = load_dataset(key, progress=progress)
    return _LOADED[key]


def split_of(ds, split):
    """(X, y) du split demandé — évite de répéter le ternaire partout."""
    return (ds.X_train, ds.y_train) if split == "Train" else (ds.X_test, ds.y_test)


def label_name(ds, y):
    """Nom lisible d'un label : « 7 » pour MNIST, « cat » pour Quick, Draw!."""
    return ds.class_names[int(y)]


# ------------------------------------------------------------------ Helpers


def format_bytes(n_bytes):
    if n_bytes >= 1024 ** 3:
        return f"{n_bytes / 1024 ** 3:.1f} Go"
    return f"{n_bytes / 1024 ** 2:.0f} Mo"


def memory_note(n, k):
    """Estime la RAM du calcul de distances — le poste dominant de l'entraînement.

    compute_squared_distances passe par l'identité ||x-c||² et n'alloue plus que
    (n, k) : le produit matriciel somme les 784 pixels au passage. Le coût est
    donc devenu modeste, mais reste le produit n×k — et ni n ni K ne sont
    plafonnés dans l'UI, d'où ce garde-fou.
    """
    n, k = int(n or 0), int(k or 0)
    cost = n * k * 4
    msg = (
        f"Matrice de distances `({n}, {k})` : **~{format_bytes(cost)}** de RAM "
        f"(+ {format_bytes(n * IMAGE_DIM * 4)} pour les images elles-mêmes)."
    )

    if cost > 4 * 1024 ** 3:
        return f"🔴 {msg}\n\nn×K est énorme : risque de saturer la RAM. Baisse K ou les images."
    if cost > 1024 ** 3:
        return f"🟠 {msg}\n\nAu-delà du Go : surveille ta RAM."
    if cost > 256 * 1024 ** 2:
        return f"🟡 {msg}"
    return f"⚪ {msg}"


def released(fig):
    """Retire la figure du gestionnaire global de pyplot avant de la rendre.

    pyplot garde une référence sur chaque figure créée : sans ça, aucune n'est
    jamais collectée et la RAM grimpe à chaque interaction. La figure reste
    parfaitement affichable par Gradio après fermeture.
    """
    plt.close(fig)
    return fig


NO_MODEL = "⚠️ Sélectionne ou entraîne un modèle d'abord."


def models_for(ds_key):
    """Modèles de cet algo entraînés sur ce dataset.

    Un K-means MNIST et un K-means Quick, Draw! ont tous deux des centroïdes
    (k, 784) : les croiser « marcherait » silencieusement tout en produisant
    n'importe quoi. Même problème à venir entre algos. On filtre donc sur les
    deux critères plutôt que de tout lister.
    """
    out = []
    for name in list_models():
        try:
            _, meta = load_model(name)
        except Exception:
            continue  # .npz corrompu ou d'un format plus ancien : on l'ignore
        # Les modèles d'avant ces clés n'en ont aucune : ils ont tous été
        # entraînés au K-means sur MNIST, d'où ces valeurs par défaut.
        if meta.get("dataset", "mnist") == ds_key and meta.get("algo", "kmeans") == ALGO:
            out.append(name)
    return out


def model_choices(ds_key, selected=None):
    models = models_for(ds_key)
    if selected is None or selected not in models:
        selected = models[0] if models else None
    return gr.update(choices=models, value=selected), selected


def describe_model(name):
    if not name:
        return "ℹ️ Aucun modèle pour ce dataset. Entraîne-en un dans l'onglet **Entraînement**."

    _, meta = load_model(name)
    return (
        f"### `{name}`\n"
        f"| | |\n|---|---|\n"
        f"| **Algo** | {meta.get('algo', 'kmeans')} |\n"
        f"| **Dataset** | {meta.get('dataset', 'mnist')} |\n"
        f"| **K — clusters** | {meta['k']} |\n"
        f"| **Images d'entraînement** | {meta['n_samples']} |\n"
        f"| **Inertie finale** | {meta['inertia']:.1f} |\n\n"
        f"`max_iter={meta['max_iter']}` · `tol={meta['tolerance']:.0e}` · "
        f"`seed={meta['seed']}`"
    )


# --------------------------------------------------------------- Événements


def max_train_images(ds_key):
    """Nombre max d'images d'entraînement du dataset actif."""
    return len(get_dataset(ds_key).X_train)


def switch_dataset(ds_key, n_cur, nviz_cur, progress=gr.Progress()):
    """Change le dataset actif : recharge, refiltre les modèles, réajuste l'UI."""
    ds = get_dataset(ds_key, progress=progress)
    n_train, n_test = len(ds.X_train), len(ds.X_test)

    update, selected = model_choices(ds_key)
    banner = (
        f"**{DATASETS[ds_key]}**  \n{n_train} images d'entraînement · {n_test} de test · "
        f"{len(ds.class_names)} classes : {', '.join(ds.class_names)}"
    )

    # Relever le plafond ne suffit pas : 60 000 saisi sur MNIST resterait affiché
    # sur Quick, Draw! (25 715 max) et ne planterait qu'au clic sur Entraîner.
    # On ramène donc aussi les valeurs dans les nouvelles bornes.
    clamp = lambda v, hi: min(int(v or 1), hi)

    return (
        banner,
        update,
        describe_model(selected),
        # Les bornes dépendent du dataset : elles doivent suivre.
        gr.update(
            maximum=n_train,
            value=clamp(n_cur, n_train),
            label=f"Images d'entraînement (max {n_train})",
        ),
        gr.update(
            maximum=n_train,
            value=clamp(nviz_cur, n_train),
            label=f"Images à projeter (max {n_train})",
        ),
        # Les figures affichées viennent de l'ancien dataset : on les vide plutôt
        # que de laisser croire qu'elles concernent le nouveau.
        None,   # codec train
        None,   # codec test
        "",
        None,   # latent train
        None,   # latent test
        None,   # distribution train
        None,   # distribution test
        "",
        None,   # centroïdes
        "",
    )


def sync_name(ds_key, k, n_samples, current, previous_auto):
    """Le nom suit dataset/K/n_samples, mais n'écrase jamais une saisie manuelle.

    k/n_samples viennent de gr.Number : vider le champ renvoie None, d'où le or 0.
    """
    auto = f"{ds_key}_k{int(k or 0)}_n{int(n_samples or 0)}"
    keep_current = current and current.strip() and current != previous_auto
    return (current if keep_current else auto), auto


def train(ds_key, k, n_samples, seed, max_iter, tolerance, name):
    name = (name or "").strip()
    if not name:
        raise gr.Error("Donne un nom au modèle avant d'entraîner.")

    ds = get_dataset(ds_key)
    k, n_samples, max_iter = int(k or 0), int(n_samples or 0), int(max_iter or 0)

    if k < 1:
        raise gr.Error("K doit valoir au moins 1.")
    if max_iter < 1:
        raise gr.Error("Il faut au moins 1 itération.")
    if not 1 <= n_samples <= len(ds.X_train):
        raise gr.Error(
            f"Images d'entraînement : entre 1 et {len(ds.X_train)} "
            f"(taille du split train de {ds_key})."
        )
    # initialize_centroids tire k indices parmi n sans remise : si k > n, elle
    # renvoie silencieusement moins de k centroïdes et fit_kmeans casse plus loin
    # sur un « slice index out of bounds » incompréhensible.
    if k > n_samples:
        raise gr.Error(
            f"K={k} dépasse le nombre d'images ({n_samples}) : impossible d'avoir plus "
            f"de clusters que de points. Baisse K ou augmente les images."
        )

    X_fit = ds.X_train[:n_samples]
    centroids, labels, history = fit_kmeans(
        X=X_fit,
        k=k,
        max_iter=max_iter,
        tolerance=float(tolerance),
        seed=int(seed),
        verbose=False,
        return_history=True,
    )
    inertia = float(compute_inertia(X_fit, labels, centroids))

    save_model(
        name,
        centroids,
        {
            "algo": ALGO,
            "dataset": ds_key,
            "k": k,
            "n_samples": n_samples,
            "max_iter": max_iter,
            "tolerance": float(tolerance),
            "seed": int(seed),
            "inertia": inertia,
            # Persisté avec le modèle : sans ça, la courbe d'entraînement serait
            # perdue à la fin de fit_kmeans et introuvable au rechargement.
            "inertia_history": history,
        },
    )
    status = (
        f"✅ Modèle **{name}** entraîné sur {ds_key} et sauvegardé — inertie finale : "
        f"{inertia:.1f} après {len(history)} itération(s). Il est sélectionné ci-dessus."
    )
    update, selected = model_choices(ds_key, name)
    return update, describe_model(selected), status


def remove_model(ds_key, name):
    if not name:
        raise gr.Error("Aucun modèle sélectionné.")
    delete_model(name)
    update, selected = model_choices(ds_key)
    return update, describe_model(selected), f"🗑️ Modèle **{name}** supprimé."


def reroll_seed():
    """Nouvelle graine de tirage, pour re-piocher d'autres images."""
    return int(np.random.default_rng().integers(0, 1_000_000))


def run_codec(ds_key, name, n_images, seed):
    """Encode/décode n images tirées au hasard, une figure par split."""
    if not name:
        return None, None, NO_MODEL

    ds = get_dataset(ds_key)
    centroids, meta = load_model(name)
    n_images = int(np.clip(int(n_images or 1), 1, 30))

    figures, stats = [], {}
    for split in ("Train", "Test"):
        X_split, y_split = split_of(ds, split)

        # Même graine pour les deux splits, mais tirage indépendant : on veut des
        # images variées, pas le même index des deux côtés (ce que faisait la
        # version précédente, qui ne montrait qu'une seule image par split).
        rng = np.random.default_rng(int(seed or 0) + (0 if split == "Train" else 1))
        idx = rng.choice(len(X_split), size=min(n_images, len(X_split)), replace=False)

        originals = X_split[idx]
        codes = [int(encode(img, centroids).numpy()) for img in originals]
        reconstructions = [decode(code, centroids) for code in codes]

        stats[split] = len(set(codes))
        figures.append(
            released(
                show_reconstructions(
                    originals,
                    reconstructions,
                    top_labels=[label_name(ds, y) for y in y_split[idx]],
                    bottom_labels=[f"code {c}" for c in codes],
                    title=f"{split.upper()} — {len(idx)} images tirées au hasard",
                    show=False,
                )
            )
        )

    bits = int(np.ceil(np.log2(meta["k"]))) if meta["k"] > 1 else 0
    ratio = compute_compression_ratio(meta["k"], IMAGE_DIM)
    mse = mse_from_inertia(meta["inertia"], meta["n_samples"], IMAGE_DIM)

    info = (
        f"**{n_images} images tirées au hasard** dans chaque split de **{ds_key}** "
        f"(graine {int(seed or 0)} — relance le tirage pour en voir d'autres).\n\n"
        f"Chaque image (784 pixels) est transmise sous la forme d'**un seul entier**, "
        f"soit {bits} bits pour K={meta['k']} — un taux de compression de "
        f"**{ratio:.0f}:1** face aux {IMAGE_DIM * 8} bits de l'image brute (le "
        f"dictionnaire des centroïdes n'étant transmis qu'une fois).\n\n"
        f"Le décodeur renvoie le centroïde portant ce numéro : deux images partageant "
        f"un code se reconstruisent **à l'identique** — visible dès que deux colonnes "
        f"ont le même code.\n\n"
        f"Le prix de cette compression est la **MSE : {mse:.4f}** par pixel "
        f"(mesurée sur le train à l'entraînement). C'est l'écart que tu vois entre la "
        f"ligne du haut et celle du bas.\n\n"
        f"Codes distincts sur ce tirage : {stats['Train']}/{n_images} en train, "
        f"{stats['Test']}/{n_images} en test."
    )
    return figures[0], figures[1], info


def run_latent(ds_key, name, n_viz, show_true):
    """Projette les deux splits d'un coup : nuage PCA + distribution des classes."""
    if not name:
        return None, None, None, None, NO_MODEL

    ds = get_dataset(ds_key)
    centroids, meta = load_model(name)

    n_viz = int(n_viz or 0)
    if n_viz < 1:
        raise gr.Error("Images à projeter : au moins 1.")
    if n_viz > len(ds.X_train):
        raise gr.Error(
            f"Images à projeter : au plus {len(ds.X_train)} (split train de {ds_key})."
        )

    y_label = "Chiffre réel" if ds_key == "mnist" else "Classe réelle"

    figures, distributions, counts = [], [], {}
    for split in ("Train", "Test"):
        X_all, y_all = split_of(ds, split)
        # Test est plus petit : on prend ce qui existe plutôt que de refuser.
        n = min(n_viz, len(X_all))
        counts[split] = n
        X_viz, y_viz = X_all[:n], y_all[:n]

        labels_viz = assign_clusters(compute_squared_distances(X_viz, centroids))
        figures.append(
            released(
                plot_latent_space(
                    X_viz,
                    labels_viz,
                    centroids=centroids,
                    y_true=y_viz if show_true else None,
                    title=f"{split.upper()} — {n} images (projection PCA)",
                    y_label=y_label,
                    show=False,
                )
            )
        )
        distributions.append(
            released(
                plot_class_distribution(
                    labels_viz,
                    y_viz,
                    class_names=ds.class_names,
                    k=meta["k"],
                    title=f"{split.upper()} — distribution des classes réelles par cluster",
                    show=False,
                )
            )
        )

    info = (
        f"**{ds_key} — Train : {counts['Train']} images · Test : {counts['Test']} images.** "
        f"Le modèle, lui, a été entraîné sur {meta['n_samples']} images de train.\n\n"
    )
    if counts["Test"] < counts["Train"]:
        info += f"ℹ️ Test plafonne à {len(ds.X_test)} images : c'est toute sa taille.\n\n"
    info += (
        f"Rappel : le vrai espace latent de ce K-means est **un entier discret** dans "
        f"{{0, …, {meta['k'] - 1}}}. Ces nuages sont une *projection PCA des données* en "
        "2D colorée par cluster — une vue de la structure trouvée, pas l'espace latent "
        "lui-même.\n\n"
        "⚠️ Chaque split calcule **sa propre** PCA : les axes des deux figures ne sont "
        "pas les mêmes repères. Compare les *formes* des groupes, pas les positions — "
        "un nuage peut apparaître mirroité, le signe des composantes étant arbitraire.\n\n"
        "**Distribution par cluster** : une barre par cluster, découpée en segments "
        "colorés — un par classe réelle. Une barre d'une seule couleur = cluster pur ; "
        "une barre bariolée = cluster qui mélange des classes. La hauteur donne en prime "
        "la taille du cluster, et le numéro sous chaque barre est le **code du codec** — "
        "le même que dans l'onglet Compression.\n\n"
        "Les clusters sont répartis par groupes de 20 : au-delà, les barres deviennent "
        "illisibles. La pureté indiquée est la part des images tombant dans un cluster "
        "où leur classe est majoritaire — c'est ce que l'inertie ne mesure **pas**.\n\n"
        + memory_note(n_viz, meta["k"])
    )
    return figures[0], figures[1], distributions[0], distributions[1], info


def model_rows(name):
    """Les métadonnées du modèle en lignes (libellé, valeur) — table ou markdown."""
    _, meta = load_model(name)
    hist = history_of(meta)
    rows = [
        ("Algo", meta.get("algo", "kmeans")),
        ("Dataset", meta.get("dataset", "mnist")),
        ("K — clusters", f"{meta['k']}"),
        ("Images d'entraînement", f"{meta['n_samples']}"),
        ("Inertie finale", f"{meta['inertia']:.1f}"),
    ]
    if hist:
        rows.append(("Itérations effectuées", f"{len(hist)} / {meta['max_iter']} max"))
    # La MSE plutôt que l'inertie brute : c'est la seule des deux qui se compare
    # d'un modèle à l'autre, l'inertie dépendant du nombre d'images.
    rows.append(
        ("MSE de reconstruction",
         f"{mse_from_inertia(meta['inertia'], meta['n_samples'], IMAGE_DIM):.4f}")
    )
    rows += [
        ("Tolérance", f"{meta['tolerance']:.0e}"),
        ("Seed", f"{meta['seed']}"),
    ]
    return rows


def plot_metadata_table(name, figsize=None):
    """Rend le tableau de métadonnées en image, collable dans un rapport."""
    rows = model_rows(name)
    height = 0.42 * len(rows) + 0.9
    fig, ax = plt.subplots(figsize=figsize or (6.4, height))
    ax.axis("off")

    table = ax.table(
        cellText=[[k, v] for k, v in rows],
        colWidths=[0.58, 0.42],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if col == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f5f5f5")

    ax.set_title(name, fontsize=12, fontweight="bold", pad=14, family="monospace")
    fig.tight_layout()
    return fig


def history_of(meta):
    """Inerties d'entraînement du modèle, une par itération (liste vide si absent).

    Une parenthèse a stocké un dict {"train": [...], "test": [...]} : on en
    ressort la série de train plutôt que de rejeter ces modèles.
    """
    raw = meta.get("inertia_history")
    if not raw:
        return []
    if isinstance(raw, dict):
        return raw.get("train", [])
    return raw


def run_inertia(ds_key, name):
    """Courbe d'entraînement du modèle sélectionné."""
    if not name:
        return None, NO_MODEL

    _, meta = load_model(name)
    history = history_of(meta)
    if not history:
        # Les modèles entraînés avant que l'historique soit persisté n'en ont pas :
        # il est perdu, seule l'inertie finale a survécu.
        return None, (
            f"⚠️ **{name}** a été entraîné avant l'ajout des courbes : son historique "
            f"n'a pas été sauvegardé. Seule son inertie finale est connue "
            f"({meta['inertia']:.1f}). Réentraîne-le pour obtenir sa courbe."
        )

    n_samples = meta["n_samples"]
    fig = plot_inertia(history, n_samples=n_samples, image_dim=IMAGE_DIM, show=False)

    # Séparateur de milliers à la française. Formaté à part : un .replace(",", " ")
    # sur la phrase entière avalerait aussi les virgules de ponctuation.
    fr = lambda v: f"{v:,.0f}".replace(",", " ")

    mse_start = mse_from_inertia(history[0], n_samples, IMAGE_DIM)
    mse_end = mse_from_inertia(history[-1], n_samples, IMAGE_DIM)

    info = (
        f"**{len(history)} itération(s)** — l'inertie passe de {fr(history[0])} à "
        f"{fr(history[-1])}, soit −{(1 - history[-1] / history[0]) * 100:.1f} %.\n\n"
        "L'inertie est la somme des distances au carré de chaque point à son "
        "centroïde : c'est exactement ce que K-means minimise, **sa loss**. Elle "
        "**ne peut que décroître** — une remontée signalerait un bug, pas un mauvais "
        "réglage. Le plateau final est le point fixe où `has_converged` arrête la "
        "boucle.\n\n"
        f"### MSE de reconstruction\n\n"
        f"L'axe **rouge de droite** gradue la même courbe en MSE : "
        f"{mse_start:.4f} → **{mse_end:.4f}** par pixel, soit un écart typique de "
        f"{np.sqrt(mse_end):.3f} sur une échelle 0–1.\n\n"
        f"Une seule courbe et non deux, car ce sont **les mêmes valeurs** : le codec "
        f"reconstruit chaque image par son centroïde, celui-là même dont l'inertie "
        f"mesure l'écart. D'où `MSE = inertie / (n × 784)` — ici un facteur "
        f"{fr(n_samples * IMAGE_DIM)}. Superposer les deux tracés donnerait deux "
        f"courbes rigoureusement confondues.\n\n"
        f"La MSE est la lecture utile : contrairement à l'inertie, elle ne dépend ni "
        f"du nombre d'images ni de leur taille, donc elle se compare d'un modèle à "
        f"l'autre."
    )
    return released(fig), info


def export_all(ds_key, name, n_viz, n_images, seed, progress=gr.Progress()):
    """Génère toutes les figures du modèle d'un coup, prêtes pour le rapport.

    Chaque figure part en PNG 150 dpi, plus une archive ZIP pour tout récupérer
    en un clic. Les figures sont écrites sur disque plutôt que rendues en base64 :
    c'est ce qui permet le téléchargement et le clic droit « copier l'image ».
    """
    if not name:
        raise gr.Error("Sélectionne ou entraîne un modèle d'abord.")

    ds = get_dataset(ds_key)
    centroids, meta = load_model(name)

    n_viz = int(np.clip(int(n_viz or 1), 1, len(ds.X_train)))
    n_images = int(np.clip(int(n_images or 10), 1, 30))
    seed = int(seed or 0)

    out_dir = Path(tempfile.mkdtemp(prefix="rapport_"))
    saved, skipped = [], []

    def emit(filename, fig):
        path = out_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    progress(0.05, desc="Tableau des métadonnées…")
    emit("01_metadata.png", plot_metadata_table(name))

    progress(0.15, desc="Courbe d'entraînement…")
    history = history_of(meta)
    if history:
        emit(
            "02_courbe_inertie.png",
            plot_inertia(history, n_samples=meta["n_samples"], image_dim=IMAGE_DIM,
                         show=False),
        )
    else:
        skipped.append(
            "**la courbe d'inertie** — ce modèle est antérieur à leur sauvegarde, "
            "réentraîne-le pour l'obtenir"
        )

    progress(0.3, desc="Compression / décompression…")
    codec_train, codec_test, _ = run_codec(ds_key, name, n_images, seed)
    emit("03_codec_train.png", codec_train)
    emit("04_codec_test.png", codec_test)

    progress(0.5, desc="Espaces latents et distributions…")
    lat_train, lat_test, dist_train, dist_test, _ = run_latent(ds_key, name, n_viz, True)
    emit("05_latent_train.png", lat_train)
    emit("06_latent_test.png", lat_test)
    emit("07_distribution_train.png", dist_train)
    emit("08_distribution_test.png", dist_test)

    progress(0.75, desc="Cartographie des centroïdes…")
    # plot_centroid_map trie par taille de cluster et annote la classe dominante :
    # il lui faut donc les assignations d'un échantillon, pas juste les centroïdes.
    labels_map = assign_clusters(compute_squared_distances(ds.X_train[:n_viz], centroids))
    emit(
        "09_cartographie_centroides.png",
        plot_centroid_map(
            centroids,
            cluster_labels=labels_map,
            y_true=ds.y_train[:n_viz],
            sort_by_size=True,
            title=f"Cartographie des centroïdes — {name}",
            show=False,
        ),
    )

    progress(0.95, desc="Archive ZIP…")
    zip_path = out_dir / f"{name}_rapport.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in saved:
            archive.write(path, path.name)

    info = (
        f"### {len(saved)} figures générées pour `{name}`\n\n"
        f"Télécharge le **ZIP** pour tout récupérer d'un coup, ou fais un clic droit → "
        f"« Copier l'image » sur une vignette de l'aperçu.\n\n"
        f"| Fichier | Contenu |\n|---|---|\n"
        f"| `01_metadata.png` | le tableau du modèle |\n"
        + ("| `02_courbe_inertie.png` | la courbe d'entraînement |\n" if history else "")
        + f"| `03_codec_train.png` · `04_codec_test.png` | {n_images} images "
        f"compressées/décompressées par split (graine {seed}) |\n"
        f"| `05_latent_train.png` · `06_latent_test.png` | espaces latents, {n_viz} images |\n"
        f"| `07_distribution_train.png` · `08_distribution_test.png` | distribution des "
        f"classes réelles par cluster |\n"
        f"| `09_cartographie_centroides.png` | les {meta['k']} centroïdes triés par taille |\n"
    )
    if ds_key != "mnist":
        info += (
            f"\nℹ️ Sur la cartographie, le chiffre sous chaque centroïde est l'**indice** "
            f"de la classe dominante : {', '.join(f'{i} = {c}' for i, c in enumerate(ds.class_names))}.\n"
        )
    if skipped:
        info += "\n⚠️ Non généré : " + " ; ".join(skipped) + ".\n"

    return [str(p) for p in saved], str(zip_path), info


def run_centroids(ds_key, name):
    """Affiche TOUS les centroïdes du modèle — aucune troncature."""
    if not name:
        return None, NO_MODEL

    centroids, meta = load_model(name)
    k_model = meta["k"]

    # La grille s'adapte à K, sinon 10 colonnes fixes donnent une figure de
    # 170 pouces de haut à K=1000 (23 s de rendu). En grille ~carrée le même
    # K=1000 tient en 16x16 pouces. Au-delà de 50 centroïdes les titres
    # « code i » deviennent illisibles : on les retire plutôt que d'agrandir.
    if k_model <= 50:
        n_cols = min(k_model, 10)
        titles = [f"code {i}" for i in range(k_model)]
        cell_w, cell_h = 1.5, 1.7
    else:
        n_cols = int(np.ceil(np.sqrt(k_model)))
        titles = None
        cell_w = cell_h = 0.5

    n_rows = int(np.ceil(k_model / n_cols))

    fig = show_images(
        centroids,
        titles=titles,
        n_rows=n_rows,
        n_cols=n_cols,
        figsize=(n_cols * cell_w, n_rows * cell_h),
        show=False,
    )

    note = (
        f"**Les {k_model} centroïdes du modèle**, en grille {n_rows}×{n_cols}. Chaque "
        "centroïde est l'image moyenne de son cluster. `decode(code)` renvoie exactement "
        "une de ces images : c'est tout le vocabulaire du codec."
    )
    if titles is None:
        note += (
            f"\n\n*Les numéros de code sont masqués au-delà de 50 centroïdes (illisibles "
            f"à cette taille) : ils se lisent de gauche à droite, ligne par ligne, de 0 à "
            f"{k_model - 1}.*"
        )
    return released(fig), note


# ---------------------------------------------------------------------- UI

with gr.Blocks(title="K-means codec") as demo:
    gr.Markdown("# K-means comme codec")

    _ds = get_dataset(DEFAULT_DATASET)
    _n_train, _n_test = len(_ds.X_train), len(_ds.X_test)
    _models = models_for(DEFAULT_DATASET)
    DEFAULT_NAME = f"{DEFAULT_DATASET}_k10_n1000"

    # Doit refléter la valeur initiale de name_in : sync_name compare le nom
    # courant à ce témoin pour distinguer « auto » d'« édité à la main ». Les
    # désynchroniser ferait passer le nom par défaut pour une saisie utilisateur,
    # et il ne suivrait plus jamais dataset/K/n_samples.
    auto_state = gr.State(DEFAULT_NAME)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Dataset")
            dataset_dd = gr.Dropdown(
                choices=[(label, key) for key, label in DATASETS.items()],
                value=DEFAULT_DATASET,
                label="Dataset actif (vaut pour tous les onglets)",
                interactive=True,
            )
            dataset_md = gr.Markdown(
                f"**{DATASETS[DEFAULT_DATASET]}**  \n{_n_train} images d'entraînement · "
                f"{_n_test} de test · {len(_ds.class_names)} classes : "
                f"{', '.join(_ds.class_names)}"
            )

            gr.Markdown("### Modèle actif")
            model_dd = gr.Dropdown(
                choices=_models,
                value=_models[0] if _models else None,
                label="Modèle entraîné (filtré par dataset)",
                interactive=True,
            )
            delete_btn = gr.Button("Supprimer ce modèle", variant="stop", size="sm")
            model_md = gr.Markdown(describe_model(_models[0] if _models else None))
            status_md = gr.Markdown()

        with gr.Column(scale=3):
            with gr.Tabs():
                # ----------------------------------------- Entraînement
                with gr.Tab("Entraînement"):
                    gr.Markdown("## Entraîner un K-means")

                    # Champs libres et non des sliders : aucun plafond arbitraire
                    # sur K ni sur les itérations, et le nombre d'images monte
                    # jusqu'à la taille réelle du split train.
                    with gr.Row():
                        k_in = gr.Number(
                            value=10,
                            precision=0,
                            minimum=1,
                            label="K — nombre de clusters (sans plafond)",
                        )
                        with gr.Column(min_width=170):
                            n_in = gr.Number(
                                value=1000,
                                precision=0,
                                minimum=1,
                                maximum=_n_train,
                                label=f"Images d'entraînement (max {_n_train})",
                            )
                            max_train_btn = gr.Button("Tout le split train", size="sm")
                        seed_in = gr.Number(value=42, precision=0, label="Seed")

                    with gr.Row():
                        iter_in = gr.Number(
                            value=100,
                            precision=0,
                            minimum=1,
                            label="Itérations max (sans plafond)",
                        )
                        tol_in = gr.Dropdown(
                            choices=[("1e-6", 1e-6), ("1e-5", 1e-5), ("1e-4", 1e-4), ("1e-3", 1e-3)],
                            value=1e-4,
                            label="Tolérance de convergence",
                        )

                    mem_md = gr.Markdown(memory_note(1000, 10))
                    name_in = gr.Textbox(value=DEFAULT_NAME, label="Nom du modèle")
                    train_btn = gr.Button("Entraîner", variant="primary")

                # ------------------------- Compression / Décompression
                with gr.Tab("Compression / Décompression") as tab_codec:
                    gr.Markdown("## Compression / Décompression")
                    gr.Markdown(
                        "Des images tirées **au hasard**, encodées puis décodées, sur les "
                        "**deux splits à la fois**. Originaux en haut de chaque figure, "
                        "reconstructions en dessous."
                    )

                    with gr.Row():
                        codec_n_in = gr.Number(
                            value=10, precision=0, minimum=1, maximum=30,
                            label="Images par split (max 30)",
                        )
                        codec_seed_in = gr.Number(
                            value=0, precision=0, label="Graine du tirage",
                        )
                        codec_reroll_btn = gr.Button("Nouveau tirage aléatoire")

                    codec_plot_train = gr.Plot(label="TRAIN")
                    codec_plot_test = gr.Plot(label="TEST")
                    codec_md = gr.Markdown()

                # ----------------------------------------- Espace latent
                with gr.Tab("Espace latent") as tab_latent:
                    gr.Markdown("## Espace latent")
                    gr.Markdown(
                        "Les **deux splits projetés en une fois**. Test étant plus petit, "
                        "il plafonne à sa taille."
                    )

                    with gr.Row():
                        with gr.Column(min_width=170):
                            nviz_in = gr.Number(
                                value=1000,
                                precision=0,
                                minimum=1,
                                maximum=_n_train,
                                label=f"Images à projeter (max {_n_train})",
                            )
                            max_viz_btn = gr.Button("Tout le split train", size="sm")
                        true_in = gr.Checkbox(value=True, label="Comparer aux classes réelles")

                    latent_btn = gr.Button("Projeter train et test", variant="primary")
                    latent_plot_train = gr.Plot(label="TRAIN — projection PCA")
                    latent_plot_test = gr.Plot(label="TEST — projection PCA")
                    gr.Markdown("### Distribution des classes réelles par cluster")
                    latent_dist_train = gr.Plot(label="TRAIN — distribution")
                    latent_dist_test = gr.Plot(label="TEST — distribution")
                    latent_md = gr.Markdown()

                # -------------------------------------------- Centroïdes
                with gr.Tab("Centroïdes") as tab_centroids:
                    gr.Markdown("## Centroïdes — le dictionnaire du codec")
                    centroids_plot = gr.Plot(label="Centroïdes")
                    centroids_md = gr.Markdown()

                # ------------------------------------ Courbe d'entraînement
                with gr.Tab("Courbe d'entraînement") as tab_inertia:
                    gr.Markdown("## Courbe d'entraînement — inertie par itération")
                    inertia_plot = gr.Plot(label="Inertie")
                    inertia_md = gr.Markdown()

                # -------------------------------------------- Export rapport
                with gr.Tab("Export / Rapport"):
                    gr.Markdown("## Export pour le rapport")
                    gr.Markdown(
                        "Génère **toutes** les figures du modèle en une fois : tableau, "
                        "courbe d'entraînement, compression train/test, espaces latents "
                        "et cartographie des centroïdes."
                    )

                    with gr.Row():
                        export_n_images = gr.Number(
                            value=10, precision=0, minimum=1, maximum=30,
                            label="Images par split (compression)",
                        )
                        export_seed = gr.Number(
                            value=0, precision=0, label="Graine du tirage",
                        )
                        export_nviz = gr.Number(
                            value=1000, precision=0, minimum=1,
                            label="Images à projeter (espaces latents)",
                        )

                    export_btn = gr.Button("Générer toutes les figures", variant="primary")
                    export_zip = gr.File(label="Archive ZIP — toutes les images")
                    export_gallery = gr.Gallery(
                        label="Aperçu — clic droit sur une image pour la copier",
                        columns=3,
                        height=420,
                    )
                    export_md = gr.Markdown()

    # ------------------------------------------------------------ Câblage

    # Les boutons « Tout le split train » remplissent le champ avec le maximum du
    # dataset actif — 60 000 sur MNIST, 25 715 sur Quick, Draw!.
    max_train_btn.click(max_train_images, dataset_dd, n_in)
    max_viz_btn.click(max_train_images, dataset_dd, nviz_in)

    dataset_dd.change(
        switch_dataset,
        [dataset_dd, n_in, nviz_in],
        [
            dataset_md,
            model_dd,
            model_md,
            n_in,
            nviz_in,
            codec_plot_train,
            codec_plot_test,
            codec_md,
            latent_plot_train,
            latent_plot_test,
            latent_dist_train,
            latent_dist_test,
            latent_md,
            centroids_plot,
            centroids_md,
        ],
    )
    dataset_dd.change(sync_name, [dataset_dd, k_in, n_in, name_in, auto_state], [name_in, auto_state])

    for widget in (k_in, n_in):
        widget.change(memory_note, [n_in, k_in], mem_md)
        widget.change(
            sync_name, [dataset_dd, k_in, n_in, name_in, auto_state], [name_in, auto_state]
        )

    train_btn.click(
        train,
        [dataset_dd, k_in, n_in, seed_in, iter_in, tol_in, name_in],
        [model_dd, model_md, status_md],
    )
    delete_btn.click(remove_model, [dataset_dd, model_dd], [model_dd, model_md, status_md])
    model_dd.change(describe_model, model_dd, model_md)

    # On ne recalcule que la vue regardée, jamais les trois autres.
    # Codec et Centroïdes sont bon marché (une image, K vignettes) : rendu
    # automatique à l'ouverture de l'onglet et à chaque changement de réglage.
    codec_inputs = [dataset_dd, model_dd, codec_n_in, codec_seed_in]
    codec_outputs = [codec_plot_train, codec_plot_test, codec_md]
    tab_codec.select(run_codec, codec_inputs, codec_outputs)
    for widget in (codec_n_in, codec_seed_in, model_dd):
        widget.change(run_codec, codec_inputs, codec_outputs)
    # Le bouton ne fait que changer la graine : le .change ci-dessus redessine.
    codec_reroll_btn.click(reroll_seed, None, codec_seed_in)

    # La projection latente n'est pas plafonnée et traite les deux splits : elle
    # peut coûter des minutes sur 60 000 images. Elle ne part donc QUE sur clic
    # explicite — un rendu auto à l'ouverture figerait l'app.
    latent_btn.click(
        run_latent,
        [dataset_dd, model_dd, nviz_in, true_in],
        [latent_plot_train, latent_plot_test, latent_dist_train, latent_dist_test, latent_md],
    )

    centroids_outputs = [centroids_plot, centroids_md]
    tab_centroids.select(run_centroids, [dataset_dd, model_dd], centroids_outputs)
    model_dd.change(run_centroids, [dataset_dd, model_dd], centroids_outputs)

    # La courbe est déjà calculée et stockée dans le .npz : la tracer ne coûte
    # rien, elle peut donc se rendre à l'ouverture de l'onglet.
    inertia_outputs = [inertia_plot, inertia_md]
    tab_inertia.select(run_inertia, [dataset_dd, model_dd], inertia_outputs)
    model_dd.change(run_inertia, [dataset_dd, model_dd], inertia_outputs)

    # L'export refait toutes les vues d'un coup : uniquement sur clic explicite.
    export_btn.click(
        export_all,
        [dataset_dd, model_dd, export_nviz, export_n_images, export_seed],
        [export_gallery, export_zip, export_md],
    )


if __name__ == "__main__":
    # Gradio 6 : le thème se passe à launch(), plus au constructeur de Blocks.
    demo.launch(theme=gr.themes.Soft(), inbrowser=True)
