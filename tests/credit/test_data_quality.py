import pandas as pd
import pytest

from validation_toolkit.credit.data_quality import controle_comptages


@pytest.fixture
def fichiers(tmp_path):
    orig = tmp_path / "o.csv"
    perf = tmp_path / "p.csv"
    pd.DataFrame({"Loan Identifier": ["a", "b", "c", "c"], "x": list("1234")}).to_csv(orig, index=False)
    pd.DataFrame({"Loan Identifier": ["a", "a", "b", "z", "z", "z"], "s": list("012345")}).to_csv(perf, index=False)
    return orig, perf


def test_comptages_connus(fichiers):
    r = controle_comptages(*fichiers, id_col="Loan Identifier")
    assert r == {
        "lignes_origination": 4,
        "prets_origination_uniques": 3,      # le doublon "c" est visible
        "lignes_performance": 6,
        "prets_performance_uniques": 3,
        "lignes_par_pret_moyenne": 2.0,
        "prets_orig_sans_perf": 1,           # c
        "prets_perf_sans_orig": 1,           # z
    }


@pytest.mark.parametrize("chunksize", [1, 2, 4, 1000])
def test_resultat_independant_de_la_taille_des_blocs(fichiers, chunksize):
    reference = controle_comptages(*fichiers, id_col="Loan Identifier")
    assert controle_comptages(*fichiers, id_col="Loan Identifier", chunksize=chunksize) == reference


def test_identifiants_lus_en_texte(tmp_path):
    orig, perf = tmp_path / "o.csv", tmp_path / "p.csv"
    orig.write_text("Loan Identifier\n007\n7\n")
    perf.write_text("Loan Identifier\n007\n")
    r = controle_comptages(orig, perf, id_col="Loan Identifier")
    assert r["prets_origination_uniques"] == 2       # "007" et "7" ne sont pas confondus
    assert r["prets_orig_sans_perf"] == 1


def test_colonne_identifiant_absente(fichiers):
    with pytest.raises(ValueError):
        controle_comptages(*fichiers, id_col="Identifiant")


def test_affichage(fichiers, capsys):
    controle_comptages(*fichiers, id_col="Loan Identifier")
    sortie = capsys.readouterr().out
    assert "lignes_performance" in sortie and "2.00" in sortie
