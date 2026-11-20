import numpy as np
import pandas as pd
from scipy import stats


def moyenne_newey_west(x, retards=None):
    """Test H0 : moyenne = 0, erreur standard robuste à l'autocorrélation (noyau de Bartlett)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        raise ValueError("au moins 3 observations nécessaires")
    if retards is None:
        retards = int(np.floor(4 * (n / 100) ** (2 / 9)))
    retards = min(retards, n - 1)

    m = x.mean()
    e = x - m
    lrv = e @ e / n
    for k in range(1, retards + 1):
        lrv += 2 * (1 - k / (retards + 1)) * (e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0) / n)
    if se == 0:
        t, p = (np.inf if m != 0 else np.nan), (0.0 if m != 0 else np.nan)
    else:
        t = m / se
        p = 2 * stats.t.sf(abs(t), n - 1)
    return {"n": n, "moyenne": m, "se": se, "t": t, "p": p, "retards": retards}


def diebold_mariano(perte_ref, perte_chall, retards=None):
    """Compare deux séries de pertes appariées par période. Moyenne > 0 : le challenger perd moins."""
    a, b = np.asarray(perte_ref, float), np.asarray(perte_chall, float)
    if a.shape != b.shape:
        raise ValueError("les deux séries doivent avoir la même longueur")
    res = moyenne_newey_west(a - b, retards)
    n = res["n"]
    if np.isfinite(res["t"]):
        res["t"] *= np.sqrt((n - 1) / n)  # correction petit échantillon (Harvey et al.)
        res["p"] = 2 * stats.t.sf(abs(res["t"]), n - 1)
    return res


def ajouter_regime(df, col_annee, frontieres, defaut="normal"):
    """frontieres : liste de (nom, annee_debut, annee_fin) ; années incluses."""
    out = df.copy()
    out["regime"] = defaut
    for nom, debut, fin in frontieres:
        out.loc[out[col_annee].between(debut, fin), "regime"] = nom
    return out


def perte_annuelle(coeur, modele, observe="observe"):
    """Erreur quadratique de calibration par année : moyenne, pondérée par n, des (observé − prédit)² sur les classes."""
    c = coeur[coeur["modele"] == modele].copy()
    c["_e2n"] = (c[observe] - c["predit"]) ** 2 * c["n"]
    g = c.groupby("date")
    return (g["_e2n"].sum() / g["n"].sum()).sort_index()
