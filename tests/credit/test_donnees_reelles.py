"""Tests sur les vrais échantillons Freddie Mac 2007 et 2015.

Lents (quelques minutes) et ignorés si les fichiers ne sont pas présents.
Lancer seulement ceux-ci :  pytest -m donnees_reelles
Les exclure :               pytest -m "not donnees_reelles"
Les chiffres attendus sont ceux relevés dans exploration.ipynb (règle de défaut v2).
"""
from pathlib import Path

import pandas as pd
import pytest

from validation_toolkit.credit.data_quality import controle_comptages
from validation_toolkit.credit.defaults import ZB_DEFAUT, dates_defaut, marquer_forbearance
from validation_toolkit.credit.loaders.freddie_io import _verifier_champs, names_from_excel

RACINE = Path(__file__).resolve().parents[2]
INTERIM = RACINE / "data" / "interim"
RAW = RACINE / "data" / "raw"
LAYOUT = RAW / "file_layout_july_2026.xlsx"
ID = "Loan Identifier"

pytestmark = pytest.mark.donnees_reelles

ATTENDU = {
    2007: dict(lignes_perf=3_015_985, defauts=8_265,
               declencheurs={"statut": 8_242, "sortie": 21, "RA": 2},
               par_annee={"2009": 2_353, "2020": 140},
               forbearance_2020=122, defauts_2020=140, taux_sans=16.11),
    2015: dict(lignes_perf=3_501_425, defauts=2_058,
               declencheurs={"statut": 2_057, "RA": 1},
               par_annee={"2016": 102, "2020": 1_115},
               forbearance_2020=1_053, defauts_2020=1_115, taux_sans=1.34),
}
STATUTS_OK = {f"{i:02d}" for i in range(100)} | {"RA", "XX"}
ZB_OK = {"", "01", "02", "03", "09", "15", "16", "96"}
COLS = [ID, "Period", "Current Loan Delinquency Status", "Zero Balance Code"]


def _fichiers(an):
    return INTERIM / f"orig_{an}.csv", INTERIM / f"perf_{an}.csv"


@pytest.fixture(scope="module", params=[2007, 2015])
def an(request):
    orig, perf = _fichiers(request.param)
    if not (orig.exists() and perf.exists()):
        pytest.skip(f"{orig.name} / {perf.name} absents de data/interim")
    return request.param


@pytest.fixture(scope="module")
def defauts_v2(an):
    _, perf = _fichiers(an)
    return marquer_forbearance(perf, dates_defaut(perf))


@pytest.fixture(scope="module")
def balayage(an):
    """Un seul passage sur la performance : valeurs observées + règle recalculée autrement."""
    _, perf = _fichiers(an)
    statuts, zbs = set(), set()
    n_periode_mauvaise = n_id_mauvais = 0
    premier = {}   # id -> premier mois où une condition de défaut est vraie
    for ch in pd.read_csv(perf, dtype=str, keep_default_na=False, usecols=COLS, chunksize=500_000):
        st, zb = ch["Current Loan Delinquency Status"], ch["Zero Balance Code"]
        statuts |= set(st.unique())
        zbs |= set(zb.unique())
        n_periode_mauvaise += (~ch["Period"].str.fullmatch(r"\d{6}")).sum()
        n_id_mauvais += (~ch[ID].str.fullmatch(r"[FA]\d{2}Q[1-4]\d{7}")).sum()
        # Réécriture indépendante : expression régulière au lieu de to_numeric
        retard90 = st.str.fullmatch(r"\d\d") & (st.where(st.str.fullmatch(r"\d\d"), "00").astype(int) >= 3)
        cond = retard90 | (st == "RA") | zb.isin(ZB_DEFAUT)
        for i, m in ch.loc[cond].groupby(ID)["Period"].min().items():
            if i not in premier or m < premier[i]:
                premier[i] = m
    return dict(statuts=statuts, zbs=zbs, n_periode_mauvaise=n_periode_mauvaise,
                n_id_mauvais=n_id_mauvais, premier=premier)


# ------------------------------------------------------------- layout / alignement
def test_layout_donne_31_et_35_colonnes_sans_doublon():
    if not LAYOUT.exists():
        pytest.skip("layout absent")
    orig = names_from_excel(LAYOUT, sheet_name=0, column=1, skip_rows=2)
    perf = names_from_excel(LAYOUT, sheet_name=1, column=1, skip_rows=2)
    assert (len(orig), len(perf)) == (31, 35)
    assert len(set(orig)) == 31 and len(set(perf)) == 35
    assert ID in orig and ID in perf
    for c in [*COLS, "Borrower Assistance Plan", "Delinquency Due to Disaster"]:
        assert c in perf


def test_valeurs_conformes_au_guide_donc_colonnes_bien_alignees(balayage):
    assert balayage["statuts"] <= STATUTS_OK, balayage["statuts"] - STATUTS_OK
    assert balayage["zbs"] <= ZB_OK, balayage["zbs"] - ZB_OK
    assert balayage["n_periode_mauvaise"] == 0
    assert balayage["n_id_mauvais"] == 0


def test_fico_origination_dans_la_plage_ou_9999(an):
    orig, _ = _fichiers(an)
    fico = pd.read_csv(orig, dtype=str, keep_default_na=False, usecols=[0]).iloc[:, 0]
    num = pd.to_numeric(fico, errors="coerce")
    assert num.notna().all()
    assert (num.between(300, 850) | (num == 9999)).all()


# ----------------------------------------------------------- txt -> csv sans perte
def test_txt_et_csv_ont_le_meme_nombre_de_lignes_et_de_champs(an):
    txt_orig = RAW / f"sample_{an}" / f"sample_orig_{an}.txt"
    txt_perf = RAW / f"sample_{an}" / f"sample_perf_{an}.txt"
    if not (txt_orig.exists() and txt_perf.exists()):
        pytest.skip("txt bruts absents")
    assert _verifier_champs(txt_orig, "|") in (31, 32)     # 32 = barre finale
    assert _verifier_champs(txt_perf, "|") in (35, 36)
    with open(txt_perf, "rb") as f:
        n_txt = sum(1 for _ in f)
    _, perf = _fichiers(an)
    assert n_txt == ATTENDU[an]["lignes_perf"]
    assert controle_comptages(*_fichiers(an), id_col=ID)["lignes_performance"] == n_txt


# ------------------------------------------------------------------ comptages
def test_controle_comptages(an):
    r = controle_comptages(*_fichiers(an), id_col=ID)
    assert r["lignes_origination"] == r["prets_origination_uniques"] == 50_000
    assert r["prets_performance_uniques"] == 50_000
    assert r["lignes_performance"] == ATTENDU[an]["lignes_perf"]
    assert r["prets_orig_sans_perf"] == r["prets_perf_sans_orig"] == 0


# --------------------------------------------------------------------- défauts
def test_chiffres_releves_dans_le_notebook(an, defauts_v2):
    att = ATTENDU[an]
    assert len(defauts_v2) == att["defauts"]
    assert defauts_v2["declencheur"].value_counts().to_dict() == att["declencheurs"]
    par_an = defauts_v2["date_defaut"].str[:4].value_counts()
    for annee, n in att["par_annee"].items():
        assert par_an[annee] == n
    d2020 = defauts_v2[defauts_v2["date_defaut"].str[:4] == "2020"]
    assert len(d2020) == att["defauts_2020"]
    assert d2020["defaut_forbearance"].sum() == att["forbearance_2020"]
    sans = (~defauts_v2["defaut_forbearance"]).sum()
    assert round(100 * sans / 50_000, 2) == att["taux_sans"]


def test_regle_recalculee_autrement_donne_exactement_les_memes_defauts(defauts_v2, balayage):
    attendu = balayage["premier"]
    obtenu = dict(zip(defauts_v2["id"], defauts_v2["date_defaut"]))
    assert set(obtenu) == set(attendu)
    assert obtenu == attendu


def test_invariants_des_defauts(an, defauts_v2):
    orig, _ = _fichiers(an)
    ids = set(pd.read_csv(orig, dtype=str, keep_default_na=False, usecols=[ID])[ID])
    assert defauts_v2["id"].is_unique
    assert set(defauts_v2["id"]) <= ids
    assert (defauts_v2["date_defaut"].str[:4] >= str(an)).all()      # pas de défaut avant l'octroi
    assert 0 < len(defauts_v2) / 50_000 < 1
