import numpy as np
import pandas as pd
import pytest

from validation_toolkit.credit.models import (
    FEATURES, auc_par_annee, modele_challenger, modele_reference, tableau_coeur, walk_forward)


def faux_panel(n_prets=3000, annees=range(2008, 2015), graine=0):
    """Panel synthétique : la PD baisse quand le FICO monte et varie avec l'année."""
    rng = np.random.default_rng(graine)
    base = pd.DataFrame({
        "id": [f"L{i}" for i in range(n_prets)],
        "fico": rng.integers(550, 820, n_prets).astype(float),
        "cltv": rng.integers(50, 100, n_prets).astype(float),
        "ltv": rng.integers(50, 100, n_prets).astype(float),
        "dti": rng.integers(15, 50, n_prets).astype(float),
        "mi_pct": 0.0, "taux": rng.normal(6.5, 0.4, n_prets), "upb": rng.normal(200000, 40000, n_prets),
        "duree": 360.0, "unites": 1.0, "emprunteurs": rng.integers(1, 3, n_prets).astype(float),
        "objet": rng.choice(["P", "C", "N"], n_prets), "occupation": "P", "type_bien": "SF",
        "primo_accedant": rng.choice(["Y", "N"], n_prets), "canal": "R",
    })
    effet_annee = dict(zip(annees, np.linspace(-1.0, 0.5, len(annees))))
    morceaux = []
    for a in annees:
        d = base.copy()
        d["annee"] = a
        d["age_mois"] = (a - 2007) * 12.0
        z = -3.0 - 0.02 * (d["fico"] - 700) + effet_annee[a]
        d["defaut"] = (rng.random(n_prets) < 1 / (1 + np.exp(-z))).astype(int)
        d["defaut_hors_forbearance"] = d["defaut"]
        morceaux.append(d)
    return pd.concat(morceaux, ignore_index=True)


@pytest.fixture(scope="module")
def panel():
    return faux_panel()


@pytest.fixture(scope="module")
def preds(panel):
    return walk_forward(panel, premiere_annee_test=2011)


def test_annees_de_test_et_colonnes(preds):
    assert sorted(preds["annee"].unique()) == [2011, 2012, 2013, 2014]
    assert {"id", "annee", "defaut", "defaut_hors_forbearance", "pd_ref", "pd_chall"} <= set(preds.columns)
    assert len(preds) == 3000 * 4
    assert preds[["pd_ref", "pd_chall"]].apply(lambda s: s.between(0, 1).all()).all()


def test_les_modeles_ont_appris_quelque_chose(preds):
    auc = auc_par_annee(preds)
    assert (auc > 0.6).all().all()


def test_aucune_fuite_d_information_depuis_l_annee_de_test_ou_le_futur(panel, preds):
    """Changer les étiquettes des années >= t ne doit pas changer les prédictions des années <= t."""
    t = 2013
    modifie = panel.copy()
    rng = np.random.default_rng(99)
    futur = modifie["annee"] >= t
    modifie.loc[futur, "defaut"] = rng.integers(0, 2, futur.sum())
    nouveau = walk_forward(modifie, premiere_annee_test=2011)
    for a in (2011, 2012, 2013):
        ref = preds[preds["annee"] == a].sort_values("id")[["pd_ref", "pd_chall"]].to_numpy()
        new = nouveau[nouveau["annee"] == a].sort_values("id")[["pd_ref", "pd_chall"]].to_numpy()
        np.testing.assert_allclose(ref, new, err_msg=f"fuite détectée pour l'année {a}")
    # contrôle de sensibilité : l'année suivante, elle, utilise ces étiquettes et doit changer
    a14 = lambda d: d[d["annee"] == 2014].sort_values("id")["pd_ref"].to_numpy()
    assert not np.allclose(a14(preds), a14(nouveau))


def test_deterministe(panel, preds):
    pd.testing.assert_frame_equal(walk_forward(panel, premiere_annee_test=2011), preds)


def test_annee_sautee_si_une_seule_classe_en_apprentissage():
    p = faux_panel(n_prets=300, annees=range(2008, 2011))
    p["defaut"] = 0
    p["defaut_hors_forbearance"] = 0
    with pytest.raises(ValueError, match="aucune année"):
        walk_forward(p, premiere_annee_test=2009)


def test_valeurs_manquantes_et_categorie_inconnue(panel):
    p = panel.copy()
    p.loc[p.sample(frac=0.1, random_state=1).index, "fico"] = np.nan
    p.loc[(p["annee"] == 2014) & (p["id"].str.endswith("7")), "canal"] = "NOUVEAU"
    out = walk_forward(p, premiere_annee_test=2013)
    assert out[["pd_ref", "pd_chall"]].notna().all().all()


def test_modeles_independants_et_features_attendues():
    assert modele_reference() is not modele_reference()
    assert len(FEATURES) == 16


# ------------------------------------------------------------- tableau du cœur
def test_tableau_coeur_effectifs_et_coherence(preds):
    t = tableau_coeur(preds, nb_classes=10)
    assert set(t["modele"]) == {"ref", "chall"} and set(t["groupe"]) == set(range(1, 11))
    for (m, a), d in t.groupby(["modele", "date"]):
        assert d["n"].sum() == 3000
        assert d["n"].max() - d["n"].min() <= 1                       # effectifs égaux à 1 près
    assert t["n_evenements"].sum() == 2 * preds["defaut"].sum()       # chaque modèle compte tous les défauts
    np.testing.assert_allclose(t["observe"], t["n_evenements"] / t["n"])


def test_tableau_coeur_classes_ordonnees_par_pd_predite(preds):
    t = tableau_coeur(preds)
    for _, d in t.groupby(["modele", "date"]):
        assert d.sort_values("groupe")["predit"].is_monotonic_increasing


def test_tableau_coeur_hors_forbearance(panel):
    p = panel.copy()
    p["defaut_hors_forbearance"] = 0
    t = tableau_coeur(walk_forward(p, premiere_annee_test=2013))
    assert t["n_evenements_hors_forbearance"].sum() == 0 and t["n_evenements"].sum() > 0


def test_auc_nan_si_une_seule_classe_dans_l_annee_de_test(panel):
    p = walk_forward(panel, premiere_annee_test=2013)
    p.loc[p["annee"] == 2014, ["defaut"]] = 0
    assert auc_par_annee(p).loc[2014].isna().all()
