import numpy as np
import pandas as pd
import pytest

from validation_toolkit.credit.lgd import contraste_regimes, lgd_par_cohorte, lgd_par_pret

COLS = ["Loan Identifier", "Period", "Current Actual UPB", "Zero Balance Removal UPB", "Actual Loss",
        "Zero Balance Effective Date"]


def ecrire(tmp_path, lignes):
    complet = [tuple(l) + ("",) * (len(COLS) - len(l)) for l in lignes]
    chemin = tmp_path / "perf.csv"
    pd.DataFrame(complet, columns=COLS).to_csv(chemin, index=False)
    return chemin


def df_def(**dates):
    return pd.DataFrame({"id": list(dates), "date_defaut": list(dates.values())})


def test_lgd_simple(tmp_path):
    p = ecrire(tmp_path, [("a", "200712", "100000"), ("a", "200801", "100000"),
                          ("a", "200901", "", "98000", "30000", "200901")])
    out = lgd_par_pret(p, ["a"], df_def(a="200801"))
    r = out.iloc[0]
    assert r["lgd"] == pytest.approx(0.30)
    assert r["lgd_upb_sortie"] == pytest.approx(30000 / 98000)
    assert r["delai_mois"] == 12 and r["annee_defaut"] == 2008


def test_gain_conserve_en_negatif(tmp_path):
    p = ecrire(tmp_path, [("a", "200801", "100000"), ("a", "200901", "", "100000", "-5000", "200901")])
    assert lgd_par_pret(p, ["a"], df_def(a="200801")).iloc[0]["lgd"] == pytest.approx(-0.05)


def test_upb_defaut_absent_donne_nan(tmp_path):
    p = ecrire(tmp_path, [("a", "200801", ""), ("a", "200901", "", "100000", "30000", "200901")])
    r = lgd_par_pret(p, ["a"], df_def(a="200801")).iloc[0]
    assert np.isnan(r["lgd"]) and r["lgd_upb_sortie"] == pytest.approx(0.30)


def test_pret_hors_echantillon_ignore(tmp_path):
    p = ecrire(tmp_path, [("a", "200801", "100"), ("a", "200901", "", "100", "10", "200901"),
                          ("b", "200801", "100"), ("b", "200901", "", "100", "50", "200901")])
    out = lgd_par_pret(p, ["a"], df_def(a="200801", b="200801"))
    assert out["id"].tolist() == ["a"]


def test_echantillon_absent_leve_erreur(tmp_path):
    p = ecrire(tmp_path, [("a", "200801", "100")])
    with pytest.raises(ValueError):
        lgd_par_pret(p, ["zzz"], df_def(a="200801"))


def test_resultat_independant_des_blocs(tmp_path):
    p = ecrire(tmp_path, [("a", "200801", "100000"), ("a", "200901", "", "98000", "30000", "200901"),
                          ("b", "200802", "50000"), ("b", "200902", "", "50000", "10000", "200902")])
    d = df_def(a="200801", b="200802")
    un = lgd_par_pret(p, ["a", "b"], d, chunksize=1).set_index("id")
    tous = lgd_par_pret(p, ["a", "b"], d).set_index("id")
    pd.testing.assert_frame_equal(un.sort_index(), tous.sort_index())


def fabrique(rows):
    return pd.DataFrame(rows, columns=["annee_defaut", "perte", "upb_defaut"]).assign(
        lgd=lambda x: x["perte"] / x["upb_defaut"], delai_mois=12)


def test_cohorte_ponderee_differe_de_la_moyenne():
    df = fabrique([(2009, 90, 100), (2009, 5, 10)])
    r = lgd_par_cohorte(df, seuil=2).iloc[0]
    assert r["lgd_ponderee"] == pytest.approx(95 / 110)
    assert r["lgd_moyenne"] == pytest.approx((0.9 + 0.5) / 2)
    assert r["concluante"]


def test_cohorte_seuil():
    assert not lgd_par_cohorte(fabrique([(2009, 1, 10)]), seuil=30).iloc[0]["concluante"]


def test_contraste_signe_et_ic():
    rng = np.random.default_rng(0)
    crise = [(2009, 40 + e, 100) for e in rng.normal(0, 5, 200)]
    normal = [(2013, 20 + e, 100) for e in rng.normal(0, 5, 200)]
    r = contraste_regimes(fabrique(crise + normal), [("crise", 2008, 2010)], "crise")
    assert r["ecart"] == pytest.approx(0.20, abs=0.02)
    assert r["ic95"][0] < r["ecart"] < r["ic95"][1] and r["ic95"][0] > 0


def test_contraste_reproductible():
    df = fabrique([(2009, 40, 100), (2009, 60, 100), (2013, 20, 100), (2013, 30, 100)])
    a = contraste_regimes(df, [("crise", 2008, 2010)], "crise", nb_tirages=200, graine=1)
    b = contraste_regimes(df, [("crise", 2008, 2010)], "crise", nb_tirages=200, graine=1)
    assert a == b


def test_contraste_regime_vide():
    df = fabrique([(2013, 20, 100), (2013, 30, 100)])
    with pytest.raises(ValueError):
        contraste_regimes(df, [("crise", 2008, 2010)], "crise")
