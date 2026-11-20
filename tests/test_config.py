"""Le fichier config/freddie_mac.yaml doit rester cohérent avec les constantes du code."""
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from validation_toolkit.credit import models
from validation_toolkit.credit.defaults import SEUIL_STATUT, ZB_DEFAUT

CHEMIN = Path(__file__).resolve().parents[1] / "config" / "freddie_mac.yaml"


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(CHEMIN.read_text(encoding="utf-8"))


def test_le_yaml_se_charge_et_plus_rien_n_est_a_definir(cfg):
    assert "a_definir" not in CHEMIN.read_text(encoding="utf-8").replace("a_definir_", "")
    assert cfg["regle_defaut"]["version"] == "v2"


def test_regle_de_defaut_alignee_sur_le_code(cfg):
    r = cfg["regle_defaut"]
    assert set(r["zero_balance_codes_defaut"]) == ZB_DEFAUT
    assert r["statut_impayes_minimum"] == SEUIL_STATUT


def test_variables_alignees_sur_le_code(cfg):
    assert cfg["panel"]["variables_numeriques"] == models.NUM
    assert cfg["panel"]["variables_categorielles"] == models.CAT


def test_hyperparametres_du_challenger_alignes(cfg):
    gb = models.modele_challenger().named_steps["clf"]
    c = cfg["modeles"]["challenger"]
    assert (gb.max_iter, gb.learning_rate, gb.max_leaf_nodes, gb.min_samples_leaf, gb.random_state) == \
        (c["max_iter"], c["learning_rate"], c["max_leaf_nodes"], c["min_samples_leaf"], c["random_state"])
    assert models.modele_reference().named_steps["clf"].max_iter == cfg["modeles"]["reference"]["max_iter"]


def test_decoupes_et_classes(cfg):
    m = cfg["decoupe_temporelle"]["millesimes"]
    assert m[2007]["premiere_annee_test"] == 2011 and m[2015]["premiere_annee_test"] == 2019
    assert cfg["classes_de_risque"]["nombre"] == 10


def test_regimes_sans_chevauchement(cfg):
    f = sorted((x["annee_debut"], x["annee_fin"]) for x in cfg["regimes"]["frontieres"])
    assert all(a[1] < b[0] for a, b in zip(f, f[1:]))
