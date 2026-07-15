import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

def initialize_centroids(X, k, seed): 
    # Convertir X en tenseur TensorFlow
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    
    # Obtenir le nombre d'échantillons et la dimension
    n_samples = tf.shape(X)[0]
    
    # Sélectionner k indices aléatoires
    indices = tf.random.shuffle(tf.range(n_samples), seed=seed)[:k]
    
    # Initialiser les centroïdes avec les données correspondantes
    centroids = tf.gather(X, indices)
    
    return centroids


def compute_squared_distances(X, centroids):
    """Distances au carré entre chaque image et chaque centroïde -> (n, k).

    Calculé via l'identité ||x - c||² = ||x||² - 2·x·c + ||c||².

    La forme naïve `tf.square(X[:, None, :] - centroids)` matérialise un tenseur
    (n, k, 784) — les 784 différences pixel à pixel de chaque couple — avant de
    les sommer. Soit 784x le résultat utile : 17,5 Go pour n=60000, k=100.
    Ici le produit matriciel somme les pixels au passage, donc rien de plus gros
    que (n, k) n'est jamais alloué : 24 Mo dans le même cas.
    """
    X = tf.cast(tf.convert_to_tensor(X), tf.float32)
    centroids = tf.cast(tf.convert_to_tensor(centroids), tf.float32)

    x_squared = tf.reduce_sum(tf.square(X), axis=1, keepdims=True)   # (n, 1)
    c_squared = tf.reduce_sum(tf.square(centroids), axis=1)          # (k,)
    cross = tf.matmul(X, centroids, transpose_b=True)                # (n, k)

    distances = x_squared - 2.0 * cross + c_squared

    # Soustraire deux grands nombres proches perd des décimales : là où la
    # distance vaut zéro, le résultat peut sortir légèrement négatif (~-1e-5).
    # La forme naïve, elle, somme des carrés et ne peut jamais l'être — on
    # rétablit cette garantie, dont argmin et l'inertie dépendent.
    return tf.maximum(distances, 0.0)


def assign_clusters(distances): 
    labels = tf.argmin(distances, axis=1)
    return labels

def update_centroids(X, labels, k, old_centroids, seed=None):
    X = tf.cast(tf.convert_to_tensor(X), tf.float32)
    labels = tf.cast(tf.convert_to_tensor(labels), tf.int32)
    old_centroids = tf.cast(tf.convert_to_tensor(old_centroids), tf.float32)

    new_centroids = []

    for i in range(k):
        cluster_points = tf.boolean_mask(X, labels == i)

        if tf.shape(cluster_points)[0] > 0:
            new_centroid = tf.reduce_mean(cluster_points, axis=0)
        else:
            new_centroid = old_centroids[i]

        new_centroids.append(new_centroid)

    return tf.stack(new_centroids, axis=0)

def compute_inertia(X, labels, centroids):
    X = tf.cast(tf.convert_to_tensor(X), tf.float32)
    labels = tf.cast(tf.convert_to_tensor(labels), tf.int32)
    centroids = tf.cast(tf.convert_to_tensor(centroids), tf.float32)

    distances = compute_squared_distances(X, centroids)

    indices = tf.stack(
        [tf.range(tf.shape(X)[0], dtype=tf.int32), labels],
        axis=1
    )

    point_distances = tf.gather_nd(distances, indices)
    return tf.reduce_sum(point_distances)


def has_converged(old_centroids, new_centroids, tolerance):
    old_centroids = tf.cast(tf.convert_to_tensor(old_centroids), tf.float32)
    new_centroids = tf.cast(tf.convert_to_tensor(new_centroids), tf.float32)
    tolerance = tf.cast(tolerance, tf.float32)

    squared_shifts = tf.reduce_sum(
        tf.square(new_centroids - old_centroids),
        axis=1
    )

    return tf.reduce_all(squared_shifts < tolerance)

def fit_kmeans(X, k, max_iter, tolerance, seed, verbose, return_history=False):
    """Entraîne un K-means. Retourne (centroids, labels).

    return_history=True ajoute un 3e élément : l'inertie à chaque itération,
    c'est-à-dire la courbe d'entraînement. Elle est déjà calculée par la boucle
    (elle sert au suivi verbose) — on ne fait que la conserver. Le défaut False
    garde la signature d'origine pour les appels à deux valeurs de retour.
    """
    # Convertir X en tenseur TensorFlow
    X = tf.cast(tf.convert_to_tensor(X), tf.float32)

    # Initialiser les centroïdes
    centroids = initialize_centroids(X, k, seed)

    history = []

    # Boucle d'itération
    for iteration in range(max_iter):
        # Calculer les distances
        distances = compute_squared_distances(X, centroids)
        
        # Assignation des clusters
        labels = assign_clusters(distances)
        
        # Mise à jour des centroïdes
        new_centroids = update_centroids(X, labels, k, centroids, seed)
        
        # Calcul de l'inertie
        inertia = compute_inertia(X, labels, new_centroids)
        history.append(float(inertia))

        # Affichage optionnel
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: inertia = {inertia:.4f}")
        
        # Vérifier la convergence
        converged = has_converged(centroids, new_centroids, tolerance)
        centroids = new_centroids

        if converged:
            if verbose:
                print(f"Convergence après {iteration + 1} itérations")
            break

    if return_history:
        return centroids, labels, history
    return centroids, labels