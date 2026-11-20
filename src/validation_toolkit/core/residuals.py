"""Résidus et biais par groupe. 

Convention : résidu = observé - prédit.
Un résidu positif signifie que le modèle est en dessous de l'observé (il sous-estime).
"""
import numpy as np
import pandas as pd


def _verifier_colonnes(df, colonnes):
    manquantes = [c for c in colonnes if c not in df.columns]
    if manquantes:
        raise KeyError(f"colonnes absentes : {manquantes}")


def ajouter_residus(df, observe="observe", predit="predit", residu="residu"):
    """Retourne une copie de df avec la colonne `residu` = observé - prédit."""
    _verifier_colonnes(df, [observe, predit])
    out = df.copy()
    out[residu] = out[observe] - out[predit]
    return out


def residus_par_groupe(df, groupe="groupe", observe="observe", predit="predit"):
    """Statistiques des résidus par groupe, une ligne par groupe :

    n                   nombre de résidus utilisables (lignes sans valeur manquante)
    biais_moyen         moyenne des résidus (le signe compte : la RMSE ne le montre pas)
    rmse                racine de la moyenne des carrés des résidus
    part_positifs       part des résidus strictement positifs
    """
    _verifier_colonnes(df, [groupe, observe, predit])
    r = (df[observe] - df[predit]).rename("residu")
    r = r[r.notna() & df[groupe].notna()]
    g = r.groupby(df.loc[r.index, groupe], observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "biais_moyen": g.mean(),
        "rmse": g.apply(lambda x: np.sqrt((x ** 2).mean())),
        "part_positifs": g.apply(lambda x: (x > 0).mean()),
    })