"""App Streamlit — explorer le K-means comme codec (quantification vectorielle) sur MNIST.

Lancement, depuis src/kmeas/ :
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="K-means codec — MNIST", layout="wide")
st.title("K-means comme codec — MNIST")

# `import tensorflow` prend ~20 s. Streamlit n'affiche rien tant que le script
# n'a pas produit son premier élément : si on importait en tête de fichier,
# l'utilisateur resterait devant un écran de chargement vide sans savoir
# pourquoi. On rend donc le titre d'abord, puis on importe sous un spinner.
with st.spinner("Premier lancement : chargement de TensorFlow (~20 s)…"):
    import matplotlib.pyplot as plt
    import numpy as np

    from utils.codec import decode, encode
    from utils.data import flatten_images, load_mnist, normalize_images
    from utils.kmeas import (
        assign_clusters,
        compute_inertia,
        compute_squared_distances,
        fit_kmeans,
    )
    from utils.registry import delete_model, list_models, load_model, save_model
    from utils.visualization import plot_latent_space, show_images

    # TODO: une fois utils/metrics.py implémenté, le brancher ici pour afficher
    # MSE de reconstruction et taux de compression dans l'onglet Compression :
    #   from utils.metrics import mean_squared_error, compute_compression_ratio

IMAGE_DIM = 784


# cache_resource et non cache_data : cache_data recopie sa valeur de retour à
# chaque appel (209 Mo pour MNIST), alors qu'on ne modifie jamais ces tableaux.
@st.cache_resource(show_spinner="Chargement de MNIST…")
def get_data():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    X_train = flatten_images(normalize_images(x_train))
    X_test = flatten_images(normalize_images(x_test))
    return X_train, np.asarray(y_train), X_test, np.asarray(y_test)


def format_bytes(n_bytes):
    if n_bytes >= 1024 ** 3:
        return f"{n_bytes / 1024 ** 3:.1f} Go"
    return f"{n_bytes / 1024 ** 2:.0f} Mo"


def memory_note(n, k):
    """compute_squared_distances alloue un tenseur (n, k, 784) : ça monte vite."""
    cost = n * k * IMAGE_DIM * 4
    msg = f"Tenseur de distances intermédiaire : ~{format_bytes(cost)} en RAM."

    if cost > 2 * 1024 ** 3:
        st.error(f"{msg} Risque de saturer la RAM — baisse le nombre d'images ou K.")
    elif cost > 512 * 1024 ** 2:
        st.warning(msg)
    else:
        st.caption(msg)


def render(fig):
    """Affiche une figure matplotlib puis la ferme (sinon fuite mémoire entre reruns)."""
    st.pyplot(fig)
    plt.close(fig)


X_train, y_train, X_test, y_test = get_data()

# ------------------------------------------------------------------ Sidebar

st.sidebar.title("Modèle actif")

models = list_models()
centroids, meta, model_name = None, None, None

if models:
    model_name = st.sidebar.selectbox("Modèle entraîné", models)
    centroids, meta = load_model(model_name)

    st.sidebar.metric("K — clusters", meta["k"])
    st.sidebar.metric("Images d'entraînement", meta["n_samples"])
    st.sidebar.metric("Inertie finale", f"{meta['inertia']:.1f}")
    st.sidebar.caption(
        f"max_iter={meta['max_iter']} · tol={meta['tolerance']:.0e} · seed={meta['seed']}"
    )

    if st.sidebar.button("Supprimer ce modèle"):
        delete_model(model_name)
        st.rerun()
else:
    st.sidebar.info("Aucun modèle. Entraîne-en un dans l'onglet **Entraînement**.")

tab_train, tab_codec, tab_latent, tab_centroids = st.tabs(
    ["Entraînement", "Compression / Décompression", "Espace latent", "Centroïdes"]
)

# ------------------------------------------------------------- Entraînement

with tab_train:
    if "last_trained" in st.session_state:
        done_name, done_inertia = st.session_state.pop("last_trained")
        st.success(
            f"Modèle **{done_name}** entraîné et sauvegardé "
            f"(inertie finale : {done_inertia:.1f}). Sélectionne-le dans la barre latérale."
        )

    st.header("Entraîner un K-means")

    c1, c2, c3 = st.columns(3)
    with c1:
        k = st.slider("K — nombre de clusters", 2, 64, 10)
    with c2:
        n_samples = st.select_slider(
            "Images d'entraînement",
            options=[500, 1000, 2000, 5000, 10000],
            value=1000,
        )
    with c3:
        seed = st.number_input("Seed", value=42, step=1)

    c4, c5 = st.columns(2)
    with c4:
        max_iter = st.slider("Itérations max", 10, 500, 100)
    with c5:
        tolerance = st.select_slider(
            "Tolérance de convergence",
            options=[1e-6, 1e-5, 1e-4, 1e-3],
            value=1e-4,
            format_func=lambda t: f"{t:.0e}",
        )

    memory_note(n_samples, k)

    # Le nom suit K/n_samples par défaut, mais on ne l'écrase jamais si tu l'as
    # personnalisé (passer `value=` à text_input recréerait le widget et
    # effacerait ta saisie à chaque changement de slider).
    auto_name = f"kmeans_k{k}_n{n_samples}"
    previous_auto = st.session_state.get("_auto_name")
    if previous_auto != auto_name:
        if st.session_state.get("model_name", previous_auto) == previous_auto:
            st.session_state["model_name"] = auto_name
        st.session_state["_auto_name"] = auto_name

    name = st.text_input("Nom du modèle", key="model_name")

    if st.button("Entraîner", type="primary"):
        X_fit = X_train[:n_samples]

        with st.spinner(f"Entraînement de K={k} sur {n_samples} images…"):
            new_centroids, labels = fit_kmeans(
                X=X_fit,
                k=k,
                max_iter=max_iter,
                tolerance=tolerance,
                seed=int(seed),
                verbose=False,
            )
            inertia = float(compute_inertia(X_fit, labels, new_centroids))

        save_model(
            name,
            new_centroids,
            {
                "k": int(k),
                "n_samples": int(n_samples),
                "max_iter": int(max_iter),
                "tolerance": float(tolerance),
                "seed": int(seed),
                "inertia": inertia,
            },
        )
        st.session_state["last_trained"] = (name, inertia)
        st.rerun()

# ------------------------------------------------- Compression / Décompression

with tab_codec:
    st.header("Compression / Décompression")

    if centroids is None:
        st.warning("Sélectionne ou entraîne un modèle d'abord.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            split = st.radio("Jeu de données", ["Test", "Train"], horizontal=True)
        X_split = X_test if split == "Test" else X_train
        y_split = y_test if split == "Test" else y_train
        with c2:
            idx = st.number_input(
                "Index de l'image", min_value=0, max_value=len(X_split) - 1, value=10, step=1
            )

        image = X_split[int(idx)]
        code = encode(image, centroids)
        reconstruction = decode(code, centroids)
        code_int = int(code.numpy())

        render(
            show_images(
                [image, reconstruction],
                titles=[
                    f"Original — chiffre réel {y_split[int(idx)]}",
                    f"Reconstruit — code {code_int}",
                ],
                n_rows=1,
                n_cols=2,
                figsize=(7, 3.5),
                show=False,
            )
        )

        bits = int(np.ceil(np.log2(meta["k"])))
        st.info(
            f"L'image (784 pixels) est transmise sous la forme d'**un seul entier : {code_int}**, "
            f"soit {bits} bits pour K={meta['k']}. Le décodeur renvoie le centroïde n°{code_int} — "
            "toutes les images de ce cluster se reconstruisent donc à l'identique."
        )

# ------------------------------------------------------------- Espace latent

with tab_latent:
    st.header("Espace latent")

    if centroids is None:
        st.warning("Sélectionne ou entraîne un modèle d'abord.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            n_viz = st.select_slider(
                "Images à projeter",
                options=[200, 500, 1000, 2000, 5000],
                value=1000,
            )
        with c2:
            show_true = st.checkbox("Comparer aux chiffres réels", value=True)

        memory_note(n_viz, meta["k"])

        X_viz, y_viz = X_train[:n_viz], y_train[:n_viz]

        with st.spinner("Assignation des clusters et projection PCA…"):
            labels_viz = assign_clusters(compute_squared_distances(X_viz, centroids))
            fig = plot_latent_space(
                X_viz,
                labels_viz,
                centroids=centroids,
                y_true=y_viz if show_true else None,
                show=False,
            )
        render(fig)

        st.info(
            f"Rappel : le vrai espace latent de ce K-means est **un entier discret** dans "
            f"{{0, …, {meta['k'] - 1}}}. Ce nuage est une *projection PCA des données* en 2D "
            "colorée par cluster — une vue de la structure trouvée, pas l'espace latent lui-même."
        )

# ---------------------------------------------------------------- Centroïdes

with tab_centroids:
    st.header("Centroïdes — le dictionnaire du codec")

    if centroids is None:
        st.warning("Sélectionne ou entraîne un modèle d'abord.")
    else:
        k_model = meta["k"]
        n_cols = min(k_model, 10)
        n_rows = int(np.ceil(k_model / n_cols))

        render(
            show_images(
                centroids,
                titles=[f"code {i}" for i in range(k_model)],
                n_rows=n_rows,
                n_cols=n_cols,
                figsize=(n_cols * 1.5, n_rows * 1.7),
                show=False,
            )
        )
        st.caption(
            "Chaque centroïde est l'image moyenne de son cluster. `decode(code)` renvoie "
            "exactement une de ces images : c'est tout le vocabulaire du codec."
        )
