import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def show_centroid(centroid, title="Centroïde reconstruit", show=True):
    image = np.asarray(tf.reshape(centroid, (28, 28)))

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")

    if show:
        plt.show()
    return fig


def show_images(images, titles=None, n_rows=1, n_cols=None, figsize=None, show=True):
    """Affiche plusieurs images 28x28 dans une seule figure (grille).

    Args:
        images:  liste/tableau d'images, chacune (784,) ou (28, 28).
        titles:  liste de titres, un par image (optionnel).
        n_rows:  nombre de lignes de la grille.
        n_cols:  nombre de colonnes ; déduit de n_rows si None.
        figsize: taille de la figure ; déduite de la grille si None.
        show:    True -> plt.show() (notebook) ; False -> retourne la figure (app).

    Exemple (4 images en colonne) :
        show_images([img1, img2, img3, img4], titles=[...], n_rows=4, n_cols=1)
    """
    # Convertit chaque image en (28, 28) numpy, quel que soit le format d'entrée
    imgs = [np.asarray(tf.reshape(img, (28, 28))) for img in images]
    n = len(imgs)

    if n_cols is None:
        n_cols = int(np.ceil(n / n_rows))
    if figsize is None:
        figsize = (n_cols * 3, n_rows * 3)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # aplatit, gère le cas d'un seul axe

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
            if titles is not None:
                ax.set_title(titles[i])
        ax.axis("off")  # masque aussi les cases vides éventuelles

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_centroid_map(centroids, cluster_labels=None, y_true=None, n_cols=None,
                      sort_by_size=True, cell_size=0.6,
                      title="Cartographie des centroïdes", show=True):
    """Affiche tous les centroïdes en grille : la carte du dictionnaire appris.

    Args:
        centroids:      (k, 784) centroïdes du K-means.
        cluster_labels: (n,) index de cluster de chaque point ; nécessaire pour
                        trier par taille de cluster et pour déduire le chiffre
                        dominant.
        y_true:         (n,) labels réels ; annote chaque case du chiffre
                        majoritaire de son cluster (nécessite cluster_labels).
        n_cols:         nombre de colonnes ; déduit en grille ~carrée si None.
        sort_by_size:   True -> centroïdes triés du cluster le plus peuplé au
                        moins peuplé (les prototypes utiles d'abord, les
                        clusters vides à la fin).
        cell_size:      taille d'une case en pouces.
        show:           True -> plt.show() (notebook) ; False -> retourne la figure (app).
    """
    C = np.asarray(tf.reshape(centroids, (-1, 28, 28)))
    k = C.shape[0]

    sizes = None
    if cluster_labels is not None:
        cluster_labels = np.asarray(cluster_labels).astype(int).ravel()
        sizes = np.bincount(cluster_labels, minlength=k)

    # Ordre d'affichage
    order = np.arange(k)
    if sort_by_size:
        if sizes is None:
            raise ValueError("sort_by_size=True nécessite cluster_labels.")
        order = np.argsort(-sizes)

    # Chiffre dominant de chaque cluster
    dominant = None
    if y_true is not None:
        if cluster_labels is None:
            raise ValueError("y_true nécessite cluster_labels pour être exploité.")
        y_true = np.asarray(y_true).astype(int).ravel()
        if y_true.shape[0] != cluster_labels.shape[0]:
            raise ValueError(
                f"y_true a {y_true.shape[0]} éléments mais cluster_labels en a "
                f"{cluster_labels.shape[0]}. Passe la même tranche."
            )
        dominant = np.full(k, -1)
        for c in range(k):
            members = y_true[cluster_labels == c]
            if members.size:
                dominant[c] = np.bincount(members, minlength=10).argmax()

    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(k)))
    n_rows = int(np.ceil(k / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell_size, n_rows * cell_size))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= k:
            continue
        c = order[i]
        empty = sizes is not None and sizes[c] == 0
        # Un cluster vide garde le centroïde de son initialisation : on le
        # grise pour ne pas le lire comme un prototype appris.
        ax.imshow(C[c], cmap="gray", vmin=0, vmax=1, alpha=0.25 if empty else 1.0)
        if dominant is not None and not empty:
            ax.text(0.5, -0.02, str(dominant[c]), transform=ax.transAxes,
                    ha="center", va="top", fontsize=cell_size * 11, color="tab:blue")

    subtitle = f"{k} centroïdes"
    if sizes is not None:
        subtitle += f" — {int((sizes == 0).sum())} cluster(s) vide(s)"
        if sort_by_size:
            subtitle += ", triés par taille décroissante"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11)
    # tight_layout réserve la place des titres de chaque case et disperse la
    # grille : on règle l'espacement à la main pour garder une carte dense.
    top = 1 - 0.6 / (n_rows * cell_size)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=top,
                        wspace=0.05, hspace=0.35 if dominant is not None else 0.05)

    if show:
        plt.show()
    return fig


def _pca_2d(X, mean=None, components=None, n_iter=4, seed=0):
    """Projette X (n, d) en 2D par PCA (SVD randomisé).

    On ne veut que 2 composantes sur 784 : un SVD complet les calcule toutes
    les 784 et coûte ~100x plus cher (93 s sur 5000 images, contre 0,3 s ici)
    pour exactement le même sous-espace.

    Si mean/components sont fournis, réutilise cette projection
    (utile pour projeter les centroïdes dans le même repère que les points).
    Retourne (X_2d, mean, components).
    """
    X = np.asarray(X, dtype=np.float32)

    if mean is None or components is None:
        mean = X.mean(axis=0, keepdims=True)
        Xc = X - mean
        n, d = Xc.shape
        rank = 2 + 10  # 2 composantes + marge d'over-sampling

        if min(n, d) <= rank:
            # Matrice déjà minuscule : le SVD exact est instantané.
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        else:
            # SVD randomisé (Halko et al.) : on comprime Xc sur un petit
            # sous-espace aléatoire, puis SVD exact de cette petite matrice.
            rng = np.random.default_rng(seed)
            Y = Xc @ rng.standard_normal((d, rank)).astype(np.float32)
            for _ in range(n_iter):
                # Itérations de puissance : écrasent les composantes faibles
                # pour que Y capture bien les axes dominants.
                Y, _ = np.linalg.qr(Y)
                Y = Xc @ (Xc.T @ Y)
            Q, _ = np.linalg.qr(Y)
            _, _, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)

        components = Vt[:2]  # (2, d)
    else:
        Xc = X - mean

    X_2d = Xc @ components.T  # (n, 2)
    return X_2d, mean, components


def plot_latent_space(X, cluster_labels, centroids=None, y_true=None,
                      title="Espace latent K-means (projection PCA)", figsize=(8, 7),
                      show=True, y_label="Chiffre réel"):
    """Affiche l'espace latent du K-means : données projetées en 2D (PCA),
    colorées par cluster.

    Args:
        X:              (n, d) données d'origine (images aplaties).
        cluster_labels: (n,) index de cluster de chaque point (sortie du K-means).
        centroids:      (k, d) centroïdes ; projetés et affichés si fournis.
        y_true:         (n,) labels réels ; ajoute un 2e nuage coloré par
                        label réel pour comparaison, si fourni.
        title, figsize: options d'affichage.
        show:           True -> plt.show() (notebook) ; False -> retourne la figure (app).
        y_label:        nom de ce que désignent les labels réels. « Chiffre réel »
                        pour MNIST, mais « Classe réelle » pour Quick, Draw! —
                        les catégories n'y sont pas des chiffres.
    """
    X = np.asarray(X, dtype=np.float32)
    cluster_labels = np.asarray(cluster_labels).astype(int).ravel()

    n = X.shape[0]
    if cluster_labels.shape[0] != n:
        raise ValueError(
            f"cluster_labels a {cluster_labels.shape[0]} éléments mais X en a {n}. "
            f"Passe la même tranche de données (ex. cluster_labels[:{n}])."
        )

    # Projection PCA des points (repère partagé avec les centroïdes)
    X_2d, mean, components = _pca_2d(X)

    n_plots = 2 if y_true is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0] * n_plots, figsize[1]))
    if n_plots == 1:
        axes = [axes]

    # --- Nuage coloré par cluster K-means ---
    ax = axes[0]
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels,
                         cmap="tab10", s=8, alpha=0.6)
    if centroids is not None:
        C_2d, _, _ = _pca_2d(centroids, mean=mean, components=components)
        ax.scatter(C_2d[:, 0], C_2d[:, 1], c="black", marker="X",
                   s=200, edgecolors="white", linewidths=1.5, label="Centroïdes")
        ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel("Composante principale 1")
    ax.set_ylabel("Composante principale 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")

    # --- Nuage coloré par label réel (optionnel) ---
    if y_true is not None:
        y_true = np.asarray(y_true).astype(int).ravel()
        if y_true.shape[0] != n:
            raise ValueError(
                f"y_true a {y_true.shape[0]} éléments mais X en a {n}. "
                f"Passe la même tranche (ex. y_train[:{n}])."
            )
        ax = axes[1]
        scatter_true = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true,
                                  cmap="tab10", s=8, alpha=0.6)
        ax.set_title(f"Même projection, colorée par {y_label.lower()}")
        ax.set_xlabel("Composante principale 1")
        ax.set_ylabel("Composante principale 2")
        fig.colorbar(scatter_true, ax=ax, label=y_label)

    fig.tight_layout()
    if show:
        plt.show()
    return fig