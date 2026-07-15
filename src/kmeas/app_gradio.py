"""App Gradio — explorer le K-means comme codec (quantification vectorielle).

Lancement, depuis src/kmeas/ :
    python app_gradio.py

Le dataset se choisit dans l'app (MNIST ou Quick, Draw!) et vaut pour tous les
onglets. Le chargement passe par src/data_import.py, l'entrée unique du projet :
tous les datasets en ressortent au même format (n, 784) float32 dans [0, 1].

Port de app.py (Streamlit). Différence de modèle d'exécution : Streamlit
réexécute tout le script à chaque interaction, Gradio n'appelle que la fonction
câblée sur l'événement.
"""

import sys
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
from utils.registry import delete_model, list_models, load_model, save_model  # noqa: E402
from utils.visualization import plot_latent_space, show_images  # noqa: E402

# TODO: une fois utils/metrics.py implémenté, le brancher dans l'onglet Compression
# pour afficher MSE de reconstruction et taux de compression :
#   from utils.metrics import mean_squared_error, compute_compression_ratio

IMAGE_DIM = 784
DEFAULT_DATASET = "mnist"

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
    """Modèles entraînés sur ce dataset uniquement.

    Un K-means MNIST et un K-means Quick, Draw! ont tous deux des centroïdes
    (k, 784) : les croiser « marcherait » silencieusement tout en produisant
    n'importe quoi. On filtre donc par dataset plutôt que de tout lister.
    """
    out = []
    for name in list_models():
        try:
            _, meta = load_model(name)
        except Exception:
            continue  # .npz corrompu ou d'un format plus ancien : on l'ignore
        # Les modèles d'avant le sélecteur n'ont pas de clé « dataset » : ils
        # ont tous été entraînés sur MNIST.
        if meta.get("dataset", "mnist") == ds_key:
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
    ds_key = meta.get("dataset", "mnist")
    return (
        f"### `{name}`\n"
        f"| | |\n|---|---|\n"
        f"| **Dataset** | {ds_key} |\n"
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


def switch_dataset(ds_key, n_cur, nviz_cur, idx_cur, progress=gr.Progress()):
    """Change le dataset actif : recharge, refiltre les modèles, réajuste l'UI."""
    ds = get_dataset(ds_key, progress=progress)
    n_train, n_test = len(ds.X_train), len(ds.X_test)
    idx_max = min(n_train, n_test) - 1

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
        gr.update(
            maximum=idx_max,
            value=min(int(idx_cur or 0), idx_max),
            label=f"Index de l'image (0 à {idx_max})",
        ),
        # Les figures affichées viennent de l'ancien dataset : on les vide plutôt
        # que de laisser croire qu'elles concernent le nouveau.
        None,
        "",
        None,
        None,
        "",
        None,
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
    centroids, labels = fit_kmeans(
        X=X_fit,
        k=k,
        max_iter=max_iter,
        tolerance=float(tolerance),
        seed=int(seed),
        verbose=False,
    )
    inertia = float(compute_inertia(X_fit, labels, centroids))

    save_model(
        name,
        centroids,
        {
            "dataset": ds_key,
            "k": k,
            "n_samples": n_samples,
            "max_iter": max_iter,
            "tolerance": float(tolerance),
            "seed": int(seed),
            "inertia": inertia,
        },
    )
    status = (
        f"✅ Modèle **{name}** entraîné sur {ds_key} et sauvegardé — inertie finale : "
        f"{inertia:.1f}. Il est sélectionné ci-dessus."
    )
    update, selected = model_choices(ds_key, name)
    return update, describe_model(selected), status


def remove_model(ds_key, name):
    if not name:
        raise gr.Error("Aucun modèle sélectionné.")
    delete_model(name)
    update, selected = model_choices(ds_key)
    return update, describe_model(selected), f"🗑️ Modèle **{name}** supprimé."


def run_codec(ds_key, name, idx):
    """Encode/décode la même image sur les deux splits, côte à côte."""
    if not name:
        return None, NO_MODEL

    ds = get_dataset(ds_key)
    centroids, meta = load_model(name)

    # Le même index doit être valide sur les deux splits : on borne au plus petit.
    n_max = min(len(ds.X_train), len(ds.X_test))
    idx = int(np.clip(int(idx or 0), 0, n_max - 1))

    images, titles, codes = [], [], {}
    for split in ("Train", "Test"):
        X_split, y_split = split_of(ds, split)
        image = X_split[idx]
        code = int(encode(image, centroids).numpy())
        reconstruction = decode(code, centroids)
        codes[split] = code

        images += [image, reconstruction]
        titles += [
            f"{split} — original ({label_name(ds, y_split[idx])})",
            f"{split} — reconstruit (code {code})",
        ]

    # 2 lignes (Train, Test) x 2 colonnes (original, reconstruction)
    fig = show_images(images, titles=titles, n_rows=2, n_cols=2, figsize=(7, 7.4), show=False)

    bits = int(np.ceil(np.log2(meta["k"]))) if meta["k"] > 1 else 0
    info = (
        f"Image **n°{idx}** des deux splits de **{ds_key}** (index borné à {n_max - 1}, "
        f"la taille du split test).\n\n"
        f"Chaque image (784 pixels) est transmise sous la forme d'**un seul entier** — "
        f"code {codes['Train']} pour train, code {codes['Test']} pour test — soit "
        f"{bits} bits pour K={meta['k']}. Le décodeur renvoie le centroïde portant ce "
        f"numéro : toutes les images d'un même cluster se reconstruisent à l'identique."
    )
    return released(fig), info


def run_latent(ds_key, name, n_viz, show_true):
    """Projette les deux splits d'un coup : une figure par split, titrée."""
    if not name:
        return None, None, NO_MODEL

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

    figures, counts = [], {}
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
        + memory_note(n_viz, meta["k"])
    )
    return figures[0], figures[1], info


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
                        "La même image, encodée puis décodée sur les **deux splits à la fois** : "
                        "train en haut, test en bas."
                    )

                    idx_in = gr.Number(
                        value=10,
                        precision=0,
                        minimum=0,
                        maximum=min(_n_train, _n_test) - 1,
                        label=f"Index de l'image (0 à {min(_n_train, _n_test) - 1})",
                    )

                    codec_plot = gr.Plot(label="Train et Test — original vs reconstruction")
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
                    latent_plot_train = gr.Plot(label="TRAIN")
                    latent_plot_test = gr.Plot(label="TEST")
                    latent_md = gr.Markdown()

                # -------------------------------------------- Centroïdes
                with gr.Tab("Centroïdes") as tab_centroids:
                    gr.Markdown("## Centroïdes — le dictionnaire du codec")
                    centroids_plot = gr.Plot(label="Centroïdes")
                    centroids_md = gr.Markdown()

    # ------------------------------------------------------------ Câblage

    # Les boutons « Tout le split train » remplissent le champ avec le maximum du
    # dataset actif — 60 000 sur MNIST, 25 715 sur Quick, Draw!.
    max_train_btn.click(max_train_images, dataset_dd, n_in)
    max_viz_btn.click(max_train_images, dataset_dd, nviz_in)

    dataset_dd.change(
        switch_dataset,
        [dataset_dd, n_in, nviz_in, idx_in],
        [
            dataset_md,
            model_dd,
            model_md,
            n_in,
            nviz_in,
            idx_in,
            codec_plot,
            codec_md,
            latent_plot_train,
            latent_plot_test,
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
    codec_inputs = [dataset_dd, model_dd, idx_in]
    codec_outputs = [codec_plot, codec_md]
    tab_codec.select(run_codec, codec_inputs, codec_outputs)
    for widget in (idx_in, model_dd):
        widget.change(run_codec, codec_inputs, codec_outputs)

    # La projection latente n'est pas plafonnée et traite les deux splits : elle
    # peut coûter des minutes sur 60 000 images. Elle ne part donc QUE sur clic
    # explicite — un rendu auto à l'ouverture figerait l'app.
    latent_btn.click(
        run_latent,
        [dataset_dd, model_dd, nviz_in, true_in],
        [latent_plot_train, latent_plot_test, latent_md],
    )

    centroids_outputs = [centroids_plot, centroids_md]
    tab_centroids.select(run_centroids, [dataset_dd, model_dd], centroids_outputs)
    model_dd.change(run_centroids, [dataset_dd, model_dd], centroids_outputs)


if __name__ == "__main__":
    # Gradio 6 : le thème se passe à launch(), plus au constructeur de Blocks.
    demo.launch(theme=gr.themes.Soft(), inbrowser=True)
