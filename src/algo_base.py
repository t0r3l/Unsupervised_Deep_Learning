"""L'interface que tout algo doit présenter à l'app.

--- Pourquoi cette couche existe ---

L'app ne doit rien savoir des algos hormis leur existence. Sans ça, chaque
nouvel algo obligerait à parsemer app.py de `if algo == "kmeans"` — c'est
exactement ce qu'on cherche à éviter en sortant l'app de src/kmeas/.

Chaque algo reste chez lui : src/kmeas/algo.py adapte le K-means, 
src/kohonen/algo.py le SOM. Ce fichier ne contient que le contrat commun.

L'app ne manipule donc que des objets `Algo` : elle sait entraîner, encoder,
décoder et tracer sans jamais savoir ce qu'est un centroïde ou un neurone.

--- Ajouter un algo ---

    1. Écrire une sous-classe d'Algo dans src/<mon_algo>/algo.py.
    2. L'enregistrer dans ALGOS (src/app.py) — une ligne.

Rien d'autre. L'app découvre ses hyperparamètres via `params` et construit les
widgets toute seule.

--- Le contrat ---

Un algo de ce projet est un CODEC : il apprend un dictionnaire de k prototypes
`(k, 784)` et encode une image par l'indice du plus proche. C'est ce que
suppose l'app — un autoencodeur, dont le code n'est pas un indice, demanderait
d'élargir cette interface.

Les poids sont toujours des tableaux numpy `(k, 784)` : le K-means calcule en
TensorFlow et le SOM en numpy, mais ça ne regarde pas l'app. Chaque adaptateur
convertit.
"""

from dataclasses import dataclass, field
from typing import Any


def history_of(meta):
    """Courbe d'entraînement du modèle (liste vide si absente).

    Vit ici plutôt que dans un algo : l'app en a besoin et ne doit importer aucun
    algo en particulier.

    Une parenthèse du K-means a stocké un dict {"train": [...], "test": [...]} :
    on en ressort la série de train plutôt que de rejeter ces modèles.
    """
    raw = meta.get("inertia_history")
    if not raw:
        return []
    if isinstance(raw, dict):
        return raw.get("train", [])
    return raw


@dataclass
class Param:
    """Un hyperparamètre, décrit pour que l'app construise son widget seule.

    C'est ce qui permet à l'app d'ignorer les algos : elle lit cette description
    plutôt que de coder en dur « K-means a un K et une tolérance ».
    """

    name: str                       # clé dans le dict passé à train()
    label: str                      # libellé du widget
    kind: str = "int"               # "int" | "float" | "choice"
    default: Any = 0
    minimum: Any = None
    maximum: Any = None
    choices: Any = None             # [(libellé, valeur), ...] pour kind="choice"
    info: str = ""                  # aide sous le widget
    step: Any = None                # granularité ; 1 (int) ou 0.01 (float) si None

    def __post_init__(self):
        """Refuse un défaut que le widget lui-même rejetterait.

        Les valeurs légales d'un champ numérique sont `minimum + n · step` — et
        RIEN d'autre. Un défaut hors de cette grille produit un champ né invalide,
        que le navigateur refuse de modifier : le paramètre devient inatteignable
        sans le moindre message d'erreur. C'est exactement ce qui est arrivé à
        alpha (défaut 0,1 avec minimum=0,0001 et step=1 : légal = 0,0001 / 1,0001 /
        2,0001…). Le piège est silencieux, donc il se re-tendra : autant qu'il
        casse ici, au démarrage, plutôt que dans l'UI.
        """
        if self.kind not in ("int", "float", "choice"):
            raise ValueError(f"{self.name} : kind inconnu {self.kind!r}.")
        if self.kind == "choice":
            values = [v for _, v in (self.choices or [])]
            if self.default not in values:
                raise ValueError(
                    f"{self.name} : défaut {self.default!r} absent des choix {values}."
                )
            return

        step = self.step if self.step is not None else (1 if self.kind == "int" else 0.01)
        if step <= 0:
            raise ValueError(f"{self.name} : step doit être strictement positif.")

        base = self.minimum if self.minimum is not None else 0
        offsets = (self.default - base) / step
        if abs(offsets - round(offsets)) > 1e-6:
            raise ValueError(
                f"{self.name} : défaut {self.default} inatteignable — les valeurs "
                f"légales sont {base} + n × {step}. Aligne le défaut, le minimum "
                f"ou le step."
            )
        if self.maximum is not None and self.default > self.maximum:
            raise ValueError(f"{self.name} : défaut {self.default} > maximum {self.maximum}.")


class Algo:
    """Classe de base. Voir kmeans.py / kohonen.py pour des implémentations."""

    key = ""                        # identifiant technique, ex. "kmeans"
    label = ""                      # libellé affiché
    title = ""                      # titre de l'onglet d'entraînement

    # Vocabulaire de l'algo : l'app l'utilise pour ses libellés, plutôt que de
    # dire « centroïde » à un SOM ou « neurone » à un K-means.
    dict_label = "Prototypes"       # nom du dictionnaire
    code_label = "prototype"        # nom d'une entrée du dictionnaire
    k_label = "k — nombre de prototypes"

    params: list = field(default_factory=list)

    # ------------------------------------------------------------ Entraînement

    def train(self, X, p):
        """Entraîne sur X (n, 784) avec le dict d'hyperparamètres p.

        Retourne (weights (k, 784), meta) où meta contient au moins :
            "k"               : taille du dictionnaire
            "inertia"         : inertie finale
            "inertia_history" : l'inertie au fil de l'entraînement
        plus les hyperparamètres propres à l'algo.
        """
        raise NotImplementedError

    def k_of_params(self, p):
        """Taille du dictionnaire pour ces hyperparamètres, AVANT d'entraîner.

        Sert à l'estimation mémoire et aux garde-fous de l'UI.
        """
        raise NotImplementedError

    def check(self, p, n_samples):
        """Lève ValueError si ces hyperparamètres sont invalides. Message pour l'UI."""
        return None

    def auto_name(self, ds_key, p):
        """Nom de modèle proposé par défaut."""
        raise NotImplementedError

    # ------------------------------------------------------------------ Codec

    def assign(self, X, weights):
        """(n,) indice du prototype le plus proche de chaque image."""
        raise NotImplementedError

    def encode(self, image, weights):
        """Code d'une image — un int."""
        raise NotImplementedError

    def decode(self, code, weights):
        """Prototype de ce code — (784,) numpy."""
        raise NotImplementedError

    # --------------------------------------------------------------- Registre

    def list_models(self):
        raise NotImplementedError

    def load(self, name):
        """(weights, meta)."""
        raise NotImplementedError

    def save(self, name, weights, meta):
        raise NotImplementedError

    def delete(self, name):
        raise NotImplementedError

    # ------------------------------------------------------------------- Vues

    def plot_reconstructions(self, originals, reconstructions, top_labels,
                             bottom_labels, title):
        """Originaux en haut, reconstructions en bas.

        Cette vue ne regarde que des images : elle serait identique pour tous les
        algos. Elle passe quand même par l'interface, pour que chaque algo reste
        servi par SON package — sinon il faudrait un module de figures partagées
        à côté, et l'app devrait choisir laquelle appeler selon l'algo.
        """
        raise NotImplementedError

    def plot_distribution(self, codes, y_true, class_names, k, title):
        """Classes réelles par code. Générique elle aussi — voir ci-dessus."""
        raise NotImplementedError

    def plot_latent(self, X, weights, meta, y_true, class_names, y_label, title):
        raise NotImplementedError

    def plot_dictionary(self, weights, meta, labels, y_true, class_names):
        """Le dictionnaire appris, en une figure. Retourne (figure, note markdown)."""
        raise NotImplementedError

    def plot_curve(self, history, n_samples):
        """Courbe d'entraînement. Retourne (figure, note markdown)."""
        raise NotImplementedError

    def extra_figures(self, weights, meta, labels, y_true, class_names):
        """Vues propres à l'algo — [(titre, figure), ...], au plus 2.

        C'est l'échappatoire de l'interface : l'U-matrix n'a de sens que pour un
        SOM, la cartographie triée par taille que pour un K-means. Plutôt que de
        forcer tous les algos dans les mêmes vues, chacun ajoute les siennes.
        """
        return []

    def describe_rows(self, meta):
        """Métadonnées du modèle en lignes (libellé, valeur)."""
        raise NotImplementedError
