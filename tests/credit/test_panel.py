import numpy as np
import pandas as pd
import pytest

from validation_toolkit.credit.panel import construire_panel, derniere_periode, resume_panel

ORIG_COLS = ["Loan Identifier", "First Payment Date", "Classic FICO®",
             "Original Combined Loan-to-Value (CLTV)", "Original Loan-to-Value (LTV)",
             "Original Debt-to-Income (DTI) Ratio", "Mortgage Insurance Percentage (MI %)",
             "Original Interest Rate", "Original UPB", "Original Loan Term", "Number of Units",
             "Number of Borrowers", "Loan Purpose", "Occupancy Status", "Property Type",
             "First Time Homebuyer Indicator", "Channel"]


def orig_ligne(i, fp="200703", fico="700", dti="30"):
    return [i, fp, fico, "80", "80", dti, "0", "6.5", "200000", "360", "1", "2", "P", "P", "SF", "N", "R"]


def mois(debut, fin):
    """Liste de périodes YYYYMM de debut à fin inclus."""
    out, a, m = [], int(debut[:4]), int(debut[4:])
    while f"{a}{m:02d}" <= fin:
        out.append(f"{a}{m:02d}")
        m += 1
        if m == 13:
            a, m = a + 1, 1
    return out


@pytest.fixture
def jeu(tmp_path):
    orig = pd.DataFrame([
        orig_ligne("A"),                                  # jamais en défaut, observé jusqu'en 2010
        orig_ligne("B", fico="9999", dti="999"),          # défaut en 2009, valeurs spéciales
        orig_ligne("C"),                                  # défaut dans l'année d'octroi (2007)
        orig_ligne("D"),                                  # remboursé en 2008
        orig_ligne("E"),                                  # remboursé en 2007
        orig_ligne("F"),                                  # défaut en 2008 avec forbearance
    ], columns=ORIG_COLS)
    perf = [("A", p) for p in mois("200703", "201012")]
    perf += [("B", p) for p in mois("200703", "200912")]
    perf += [("C", p) for p in mois("200703", "200711")]
    perf += [("D", p) for p in mois("200703", "200806")]
    perf += [("E", p) for p in mois("200703", "200709")]
    perf += [("F", p) for p in mois("200703", "200804")]
    o, p = tmp_path / "o.csv", tmp_path / "p.csv"
    orig.to_csv(o, index=False)
    pd.DataFrame(perf, columns=["Loan Identifier", "Period"]).to_csv(p, index=False)
    df_def = pd.DataFrame({"id": ["B", "C", "F"], "date_defaut": ["200906", "200709", "200803"],
                           "declencheur": ["statut"] * 3, "defaut_forbearance": [False, False, True]})
    return o, p, df_def


def lignes(panel, i):
    return panel[panel["id"] == i].sort_values("annee")


def test_annees_presentes_pour_chaque_pret(jeu):
    panel = construire_panel(*jeu, annee_orig=2007)
    assert lignes(panel, "A")["annee"].tolist() == [2008, 2009, 2010]
    assert lignes(panel, "B")["annee"].tolist() == [2008, 2009]       # s'arrête l'année du défaut
    assert lignes(panel, "D")["annee"].tolist() == [2008]
    assert lignes(panel, "F")["annee"].tolist() == [2008]


def test_pret_en_defaut_ou_rembourse_l_annee_d_octroi_est_exclu(jeu):
    panel = construire_panel(*jeu, annee_orig=2007)
    assert "C" not in set(panel["id"]) and "E" not in set(panel["id"])


def test_cible(jeu):
    panel = construire_panel(*jeu, annee_orig=2007)
    assert lignes(panel, "B")["defaut"].tolist() == [0, 1]
    assert lignes(panel, "A")["defaut"].sum() == 0
    assert lignes(panel, "D")["defaut"].sum() == 0
    assert panel["defaut"].sum() == 2                      # B (2009) et F (2008)


def test_variante_hors_forbearance(jeu):
    panel = construire_panel(*jeu, annee_orig=2007)
    assert lignes(panel, "F")["defaut"].tolist() == [1]
    assert lignes(panel, "F")["defaut_hors_forbearance"].tolist() == [0]
    assert lignes(panel, "B")["defaut_hors_forbearance"].tolist() == [0, 1]


def test_valeurs_speciales_deviennent_manquantes(jeu):
    panel = construire_panel(*jeu, annee_orig=2007)
    b, a = lignes(panel, "B"), lignes(panel, "A")
    assert b["fico"].isna().all() and b["dti"].isna().all()
    assert (a["fico"] == 700).all() and (a["dti"] == 30).all()


def test_age_au_premier_janvier(jeu):
    panel = construire_panel(*jeu, annee_orig=2007)
    # premier paiement 200703 -> au 1er janvier 2008 : 10 mois, puis +12 par an
    assert lignes(panel, "A")["age_mois"].tolist() == [10, 22, 34]


def test_age_jamais_negatif(tmp_path):
    orig = pd.DataFrame([orig_ligne("Z", fp="200802")], columns=ORIG_COLS)   # premier paiement en 2008
    perf = pd.DataFrame([("Z", p) for p in mois("200802", "200912")], columns=["Loan Identifier", "Period"])
    o, p = tmp_path / "o.csv", tmp_path / "p.csv"
    orig.to_csv(o, index=False); perf.to_csv(p, index=False)
    df_def = pd.DataFrame({"id": [], "date_defaut": [], "declencheur": [], "defaut_forbearance": []})
    panel = construire_panel(o, p, df_def, annee_orig=2007)
    assert (panel["age_mois"] >= 0).all()


@pytest.mark.parametrize("chunksize", [1, 7, 1000])
def test_independant_de_la_taille_des_blocs(jeu, chunksize):
    ref = construire_panel(*jeu, annee_orig=2007)
    pd.testing.assert_frame_equal(construire_panel(*jeu, annee_orig=2007, chunksize=chunksize), ref)


def test_derniere_periode(jeu):
    d = derniere_periode(jeu[1], chunksize=5)
    assert d["A"] == "201012" and d["D"] == "200806" and d["E"] == "200709"


def test_defaut_forbearance_obligatoire(jeu):
    with pytest.raises(KeyError, match="defaut_forbearance"):
        construire_panel(jeu[0], jeu[1], jeu[2].drop(columns="defaut_forbearance"), annee_orig=2007)


def test_resume_panel(jeu, capsys):
    r = resume_panel(construire_panel(*jeu, annee_orig=2007))
    assert r.loc[2008, "prets"] == 4 and r.loc[2008, "defauts"] == 1       # A, B, D, F ; F en défaut
    assert r.loc[2009, "prets"] == 2 and r.loc[2009, "defauts"] == 1       # A, B ; B en défaut
