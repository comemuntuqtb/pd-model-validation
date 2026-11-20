import numpy as np
import pandas as pd
import pytest

from validation_toolkit.core.residuals import ajouter_residus, residus_par_groupe


def tableau():
    return pd.DataFrame({
        "groupe": ["A", "A", "A", "B", "B"],
        "observe": [5.0, 3.0, 4.0, 1.0, 1.0],
        "predit":  [4.0, 4.0, 4.0, 2.0, 0.0],
    })


def test_convention_observe_moins_predit():
    out = ajouter_residus(tableau())
    assert out["residu"].tolist() == [1.0, -1.0, 0.0, -1.0, 1.0]


def test_ajouter_residus_ne_modifie_pas_l_entree():
    df = tableau()
    avant = df.copy()
    ajouter_residus(df)
    pd.testing.assert_frame_equal(df, avant)


def test_valeurs_connues_par_groupe():
    r = residus_par_groupe(tableau())
    assert r.loc["A", "n"] == 3 and r.loc["B", "n"] == 2
    assert r.loc["A", "biais_moyen"] == 0.0
    assert r.loc["A", "rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert r.loc["A", "part_positifs"] == pytest.approx(1 / 3)
    assert r.loc["B", "biais_moyen"] == 0.0
    assert r.loc["B", "rmse"] == pytest.approx(1.0)


def test_le_biais_de_signe_est_visible_alors_que_la_rmse_est_identique():
    sous = pd.DataFrame({"groupe": ["S"] * 4, "observe": [2.0] * 4, "predit": [1.0] * 4})     # +1 partout
    sym = pd.DataFrame({"groupe": ["T"] * 4, "observe": [2.0, 0.0, 2.0, 0.0], "predit": [1.0] * 4})  # +1 / -1
    a, b = residus_par_groupe(sous).loc["S"], residus_par_groupe(sym).loc["T"]
    assert a["rmse"] == b["rmse"] == pytest.approx(1.0)
    assert a["biais_moyen"] == pytest.approx(1.0) and b["biais_moyen"] == pytest.approx(0.0)


def test_signe_du_biais_modele_trop_bas():
    df = pd.DataFrame({"groupe": ["x"] * 3, "observe": [5.0] * 3, "predit": [1.0] * 3})
    assert residus_par_groupe(df).loc["x", "biais_moyen"] > 0


def test_valeurs_manquantes_ignorees_et_comptees():
    df = tableau()
    df.loc[0, "observe"] = np.nan
    df.loc[3, "predit"] = np.nan
    r = residus_par_groupe(df)
    assert r.loc["A", "n"] == 2 and r.loc["B", "n"] == 1


def test_groupe_manquant_ignore():
    df = tableau()
    df.loc[4, "groupe"] = None
    assert list(residus_par_groupe(df).index) == ["A", "B"]
    assert residus_par_groupe(df).loc["B", "n"] == 1


def test_noms_de_colonnes_personnalises_et_groupes_numeriques():
    df = pd.DataFrame({"tranche": [1, 1, 2], "y": [1.0, 3.0, 5.0], "yhat": [2.0, 2.0, 2.0]})
    r = residus_par_groupe(df, groupe="tranche", observe="y", predit="yhat")
    assert r.loc[1, "biais_moyen"] == 0.0 and r.loc[2, "biais_moyen"] == 3.0


def test_ordre_des_lignes_et_index_sans_importance():
    df = tableau()
    melange = df.sample(frac=1, random_state=0)
    melange.index = [10, 20, 30, 40, 50]
    pd.testing.assert_frame_equal(residus_par_groupe(df), residus_par_groupe(melange))


def test_colonne_absente():
    with pytest.raises(KeyError, match="predit"):
        residus_par_groupe(tableau().drop(columns="predit"))


def test_tableau_vide():
    r = residus_par_groupe(tableau().iloc[0:0])
    assert len(r) == 0 and list(r.columns) == ["n", "biais_moyen", "rmse", "part_positifs"]
