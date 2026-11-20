"""Pouvoir discriminant et stabilité : fonctions indépendantes du domaine."""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def _verifier(y, score):
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    if y.shape != score.shape:
        raise ValueError("y et score doivent avoir la même longueur")
    if len(np.unique(y)) < 2:
        raise ValueError("il faut au moins un événement et un non-événement")
    return y, score


def gini(y, score):
    """Gini = 2 × AUC − 1. Un score plus élevé doit signifier un risque plus élevé."""
    y, score = _verifier(y, score)
    return 2 * roc_auc_score(y, score) - 1


def ks(y, score):
    """Écart maximal entre les distributions cumulées des scores des événements et des non-événements."""
    y, score = _verifier(y, score)
    return stats.ks_2samp(score[y == 1], score[y == 0]).statistic


def psi(reference, nouveau, nb_classes=10, eps=1e-6):
    """Population Stability Index. Classes = quantiles de `reference`.
    Repères usuels (conventions de métier, pas des seuils statistiques) : < 0,10 stable ; 0,10–0,25 vigilance ; > 0,25 dérive."""
    ref, new = np.asarray(reference, float), np.asarray(nouveau, float)
    bornes = np.unique(np.quantile(ref, np.linspace(0, 1, nb_classes + 1)[1:-1]))
    bornes = np.concatenate([[-np.inf], bornes, [np.inf]])
    p = np.histogram(ref, bins=bornes)[0] / len(ref)
    q = np.histogram(new, bins=bornes)[0] / len(new)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))
