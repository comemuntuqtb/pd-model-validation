import openpyxl
import pandas as pd
import pytest

from validation_toolkit.credit.loaders.freddie_io import names_from_excel, txt_to_csv

COLS = ["id", "statut", "vendeur", "zb", "note"]


def lire(csv):
    return pd.read_csv(csv, dtype=str, keep_default_na=False)


def ecrire_txt(tmp_path, texte, nom="f.txt"):
    p = tmp_path / nom
    p.write_text(texte, encoding="utf-8")
    return p


# ---------------------------------------------------------------- txt_to_csv
def test_les_codes_restent_du_texte(tmp_path):
    txt = ecrire_txt(tmp_path, "L1|00|BANK|01|a\nL2|RA|OTHER||b\nL3|XX|OTHER|96|c\n")
    csv, n = txt_to_csv(txt, COLS)
    df = lire(csv)
    assert n == 3
    assert list(df.columns) == COLS
    assert df["statut"].tolist() == ["00", "RA", "XX"]       # "00" garde ses zéros, "XX" n'est pas NaN
    assert df["zb"].tolist() == ["01", "", "96"]             # champ vide = chaîne vide


def test_virgule_et_guillemet_dans_un_champ(tmp_path):
    txt = ecrire_txt(tmp_path, 'L1|00|BANK, N.A.|01|a\nL2|01|O"THER|02|b\n')
    df = lire(txt_to_csv(txt, COLS)[0])
    assert df["vendeur"].tolist() == ["BANK, N.A.", 'O"THER']


def test_champ_commencant_par_un_guillemet_ne_mange_pas_les_lignes(tmp_path):
    """Un guillemet ouvrant ne doit pas fusionner les lignes suivantes."""
    txt = ecrire_txt(tmp_path, 'L1|00|"BANK|01|a\nL2|01|OTHER|02|b\nL3|02|OTHER|03|c\n')
    csv, n = txt_to_csv(txt, COLS)
    df = lire(csv)
    assert n == 3
    assert df["id"].tolist() == ["L1", "L2", "L3"]
    assert df["vendeur"].tolist()[0] == '"BANK'


def test_plusieurs_blocs_entete_une_seule_fois_et_ordre_conserve(tmp_path):
    lignes = [f"L{i}|00|V|01|n" for i in range(7)]
    txt = ecrire_txt(tmp_path, "\n".join(lignes) + "\n")
    csv, n = txt_to_csv(txt, COLS, chunksize=3)
    brut = csv.read_text().splitlines()
    assert n == 7
    assert brut[0] == ",".join(COLS)
    assert sum(1 for l in brut if l.startswith("id,")) == 1
    assert lire(csv)["id"].tolist() == [f"L{i}" for i in range(7)]


def test_csv_par_defaut_a_cote_du_txt(tmp_path):
    txt = ecrire_txt(tmp_path, "L1|00|V|01|n\n")
    csv, _ = txt_to_csv(txt, COLS)
    assert csv == txt.with_suffix(".csv") and csv.exists()


def test_reecriture_ecrase_l_ancien_csv(tmp_path):
    txt = ecrire_txt(tmp_path, "L1|00|V|01|n\nL2|00|V|01|n\n")
    csv = tmp_path / "out.csv"
    txt_to_csv(txt, COLS, csv)
    txt_to_csv(txt, COLS, csv)
    assert len(lire(csv)) == 2


def test_barre_finale_supprimee(tmp_path):
    txt = ecrire_txt(tmp_path, "L1|00|V|01|n|\nL2|01|V||m|\n")
    df = lire(txt_to_csv(txt, COLS, chunksize=1)[0])
    assert list(df.columns) == COLS and len(df) == 2


def test_colonne_finale_non_vide_est_refusee(tmp_path):
    txt = ecrire_txt(tmp_path, "L1|00|V|01|n|extra\n")
    with pytest.raises(ValueError, match="colonnes lues"):
        txt_to_csv(txt, COLS)


def test_mauvais_nombre_de_colonnes(tmp_path):
    txt = ecrire_txt(tmp_path, "L1|00|V\n")
    with pytest.raises(ValueError, match="3 colonnes lues, 5 noms fournis"):
        txt_to_csv(txt, COLS)


def test_fichier_vide(tmp_path):
    txt = ecrire_txt(tmp_path, "")
    with pytest.raises(ValueError, match="vide"):
        txt_to_csv(txt, COLS)


def test_ligne_trop_longue_quel_que_soit_le_bloc(tmp_path):
    txt = ecrire_txt(tmp_path, "L1|00|V|01|n\nL2|00|V|01|n\nL3|00|V|01|n|x|y\n")
    with pytest.raises(ValueError, match="ligne 3"):
        txt_to_csv(txt, COLS, chunksize=1)


def test_ligne_tronquee_n_est_pas_acceptee_en_silence(tmp_path):
    """Une ligne coupée (champs manquants) ne doit pas devenir des champs vides sans alerte."""
    txt = ecrire_txt(tmp_path, "L1|00|V|01|n\nL2|00|V\nL3|00|V|01|n\n")
    with pytest.raises(ValueError, match="ligne 2"):
        txt_to_csv(txt, COLS)


@pytest.mark.parametrize("chunksize", [1, 2, 500_000])
def test_ligne_tronquee_detectee_meme_loin_dans_le_fichier(tmp_path, chunksize):
    lignes = ["L%d|00|V|01|n" % i for i in range(10)] + ["L10|00|V"] + ["L11|00|V|01|n"]
    txt = ecrire_txt(tmp_path, "\n".join(lignes) + "\n")
    with pytest.raises(ValueError, match="ligne 11"):
        txt_to_csv(txt, COLS, chunksize=chunksize)


# ----------------------------------------------------------- names_from_excel
def faire_layout(chemin, lignes, feuille="orig"):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = feuille
    for ligne in lignes:
        ws.append(ligne)
    wb.save(chemin)
    return chemin


def test_names_from_excel_saute_le_titre_et_prend_la_bonne_colonne(tmp_path):
    p = faire_layout(tmp_path / "l.xlsx", [
        ["Titre du document", None], ["No", "Nom"],
        [1, "Loan Identifier"], [2, "  Period  "], [3, "Zero Balance Code"],
    ])
    assert names_from_excel(p, column=1, skip_rows=2) == ["Loan Identifier", "Period", "Zero Balance Code"]


def test_names_from_excel_ignore_les_cellules_vides(tmp_path):
    p = faire_layout(tmp_path / "l.xlsx", [[1, "a"], [2, None], [3, "b"]])
    assert names_from_excel(p, column=1) == ["a", "b"]


def test_names_from_excel_feuille_par_numero_ou_par_nom(tmp_path):
    p = tmp_path / "l.xlsx"
    wb = openpyxl.Workbook()
    wb.active.title = "orig"
    wb.active.append(["a"])
    wb.create_sheet("perf").append(["b"])
    wb.save(p)
    assert names_from_excel(p, sheet_name=1) == ["b"]
    assert names_from_excel(p, sheet_name="perf") == ["b"]


def test_un_nom_manquant_decale_les_colonnes_mais_txt_to_csv_le_detecte(tmp_path):
    """Si le layout perd un nom (cellule vide), le contrôle du nombre de colonnes doit alerter."""
    p = faire_layout(tmp_path / "l.xlsx", [["a"], [None], ["c"]])
    noms = names_from_excel(p)                        # 2 noms au lieu de 3
    txt = ecrire_txt(tmp_path, "1|2|3\n")
    with pytest.raises(ValueError, match="3 colonnes lues, 2 noms fournis"):
        txt_to_csv(txt, noms)
