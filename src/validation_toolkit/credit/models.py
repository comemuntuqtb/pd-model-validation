"""Deux modèles de PD à 12 mois (référence et challenger), entraînés en walk-forward,
et construction du tableau d'entrée du cœur de validation.

Walk-forward à fenêtre croissante : pour prédire l'année t, le modèle est entraîné sur les
années strictement antérieures à t (leurs étiquettes sont connues à la fin de t-1), puis
réentraîné pour t+1, etc. Aucune validation croisée aléatoire.

Les modèles n'ont aucune variable macroéconomique ni l'année en entrée : ils prédisent à partir
des caractéristiques du prêt et de son âge seulement. Leur PD ne suit donc pas le cycle (limite).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUM = ["age_mois", "fico", "cltv", "ltv", "dti", "mi_pct", "taux", "upb", "duree", "unites", "emprunteurs"]
CAT = ["objet", "occupation", "type_bien", "primo_accedant", "canal"]
FEATURES = NUM + CAT


def modele_reference():
    """Régression logistique : valeurs manquantes imputées par la médiane (avec indicateur),
    variables numériques centrées-réduites, variables catégorielles en indicatrices."""
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))])


def modele_challenger():
    """Gradient boosting par histogrammes : gère seul les valeurs manquantes."""
    prep = ColumnTransformer([
        ("num", "passthrough", NUM),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
    ])
    gb = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
        categorical_features=list(range(len(NUM), len(NUM) + len(CAT))),
        early_stopping=False, random_state=0)
    return Pipeline([("prep", prep), ("clf", gb)])


MODELES = {"ref": modele_reference, "chall": modele_challenger}


def walk_forward(panel, premiere_annee_test, cible="defaut", modeles=None):
    """Prédit la PD de chaque année de test avec un modèle entraîné sur les années antérieures.

    Retourne une ligne par (prêt, année de test) : id, annee, defaut, defaut_hors_forbearance,
    et une colonne pd_<nom> par modèle. Une année est sautée si l'apprentissage ne contient
    pas les deux classes.
    """
    modeles = modeles or MODELES
    morceaux = []
    for t in sorted(panel["annee"].unique()):
        if t < premiere_annee_test:
            continue
        train, test = panel[panel["annee"] < t], panel[panel["annee"] == t]
        if len(test) == 0 or train[cible].nunique() < 2:
            continue
        out = test[["id", "annee", "defaut", "defaut_hors_forbearance"]].copy()
        for nom, fabrique in modeles.items():
            m = fabrique()
            m.fit(train[FEATURES], train[cible])
            out[f"pd_{nom}"] = m.predict_proba(test[FEATURES])[:, 1]
        morceaux.append(out)
    if not morceaux:
        raise ValueError("aucune année de test exploitable")
    return pd.concat(morceaux, ignore_index=True)


def auc_par_annee(preds, cible="defaut"):
    """AUC (discrimination) par année de test et par modèle ; NaN si une seule classe."""
    colonnes = [c for c in preds.columns if c.startswith("pd_")]
    def une_annee(d):
        if d[cible].nunique() < 2:
            return pd.Series({c: np.nan for c in colonnes})
        return pd.Series({c: roc_auc_score(d[cible], d[c]) for c in colonnes})
    return preds.groupby("annee").apply(une_annee, include_groups=False)


def tableau_coeur(preds, nb_classes=10):
    """Tableau d'entrée du cœur de validation, une ligne par (année, classe, modèle).

    Les classes sont les déciles de la PD prédite par le modèle, recalculés chaque année de
    test (effectifs égaux à une unité près). Colonnes : date, groupe, modele, observe, predit,
    n, n_evenements, observe_hors_forbearance, n_evenements_hors_forbearance.
    """
    morceaux = []
    for col in [c for c in preds.columns if c.startswith("pd_")]:
        d = preds[["annee", "defaut", "defaut_hors_forbearance", col]].copy()
        rang = d.groupby("annee")[col].rank(method="first")
        taille = d.groupby("annee")[col].transform("size")
        d["groupe"] = np.ceil(rang * nb_classes / taille).astype(int)
        g = d.groupby(["annee", "groupe"]).agg(
            observe=("defaut", "mean"), predit=(col, "mean"), n=("defaut", "size"),
            n_evenements=("defaut", "sum"),
            observe_hors_forbearance=("defaut_hors_forbearance", "mean"),
            n_evenements_hors_forbearance=("defaut_hors_forbearance", "sum"),
        ).reset_index().rename(columns={"annee": "date"})
        g.insert(2, "modele", col[3:])
        morceaux.append(g)
    return pd.concat(morceaux, ignore_index=True)
