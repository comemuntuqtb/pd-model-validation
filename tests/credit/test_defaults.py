import pandas as pd
import pytest

from validation_toolkit.credit.defaults import dates_defaut, marquer_forbearance, resume_defauts


def defauts(df):
    """DataFrame -> dict {id: (date, déclencheur)} pour des assertions lisibles."""
    return {r.id: (r.date_defaut, r.declencheur) for r in df.itertuples()}


# ------------------------------------------------------- règle de défaut (v1)
def test_trois_mois_d_impaye_declenchent_le_defaut_au_bon_mois(ecrire_perf):
    p = ecrire_perf([("a", "200801", "00"), ("a", "200802", "01"), ("a", "200803", "02"),
                     ("a", "200804", "03"), ("a", "200805", "04")])
    assert defauts(dates_defaut(p)) == {"a": ("200804", "statut")}


def test_deux_mois_d_impaye_ne_sont_pas_un_defaut(ecrire_perf):
    p = ecrire_perf([("a", "200801", "00"), ("a", "200802", "01"), ("a", "200803", "02"),
                     ("a", "200804", "02"), ("a", "200805", "01"), ("a", "200806", "00", "01")])
    assert len(dates_defaut(p)) == 0


def test_le_statut_99_plafonne_est_un_defaut(ecrire_perf):
    p = ecrire_perf([("a", "200801", "99")])
    assert defauts(dates_defaut(p)) == {"a": ("200801", "statut")}


@pytest.mark.parametrize("statut", ["10", "11", "12", "20", "98"])
def test_statuts_a_deux_chiffres_comparés_comme_nombres(ecrire_perf, statut):
    """Piège : en texte "10" < "3". La comparaison doit être numérique."""
    p = ecrire_perf([("a", "200801", statut)])
    assert defauts(dates_defaut(p)) == {"a": ("200801", "statut")}


def test_ra_est_un_defaut(ecrire_perf):
    p = ecrire_perf([("a", "200801", "00"), ("a", "200802", "RA")])
    assert defauts(dates_defaut(p)) == {"a": ("200802", "RA")}


@pytest.mark.parametrize("zb", ["02", "03", "09", "15"])
def test_codes_de_sortie_de_defaut(ecrire_perf, zb):
    p = ecrire_perf([("a", "200801", "00"), ("a", "200802", "00", zb)])
    assert defauts(dates_defaut(p)) == {"a": ("200802", "sortie")}


@pytest.mark.parametrize("zb", ["01", "16", "96"])
def test_sorties_qui_ne_sont_pas_des_defauts(ecrire_perf, zb):
    p = ecrire_perf([("a", "200801", "00"), ("a", "200802", "00", zb)])
    assert len(dates_defaut(p)) == 0


@pytest.mark.parametrize("zb", ["16", "96"])
def test_16_et_96_comptent_si_le_statut_a_deja_declenche(ecrire_perf, zb):
    p = ecrire_perf([("a", "200801", "03"), ("a", "200802", "03", zb)])
    assert defauts(dates_defaut(p)) == {"a": ("200801", "statut")}


@pytest.mark.parametrize("statut", ["XX", ""])
def test_statut_inconnu_n_est_pas_un_defaut(ecrire_perf, statut):
    p = ecrire_perf([("a", "200801", statut), ("a", "200802", "00")])
    assert len(dates_defaut(p)) == 0


def test_le_defaut_est_definitif_et_garde_sa_premiere_date(ecrire_perf):
    p = ecrire_perf([("a", "200801", "03"), ("a", "200802", "00"), ("a", "200803", "00"),
                     ("a", "200804", "05"), ("a", "200805", "00", "01")])
    assert defauts(dates_defaut(p)) == {"a": ("200801", "statut")}


def test_priorite_du_declencheur_quand_plusieurs_conditions_le_meme_mois(ecrire_perf):
    p = ecrire_perf([("a", "200801", "03", "03"),      # statut + sortie -> statut
                     ("b", "200801", "RA", "09")])     # RA + sortie -> RA
    assert defauts(dates_defaut(p)) == {"a": ("200801", "statut"), "b": ("200801", "RA")}


def test_ordre_des_lignes_et_passage_d_annee(ecrire_perf):
    """Lignes en désordre ; 200912 est avant 201001."""
    p = ecrire_perf([("a", "201001", "04"), ("a", "200912", "03"), ("a", "200911", "00")])
    assert defauts(dates_defaut(p)) == {"a": ("200912", "statut")}


def test_plusieurs_prets_independants(ecrire_perf):
    p = ecrire_perf([("a", "200801", "03"), ("b", "200801", "00"), ("b", "200802", "00", "01"),
                     ("c", "200901", "00"), ("c", "200902", "RA"), ("d", "200801", "00", "02")])
    assert defauts(dates_defaut(p)) == {
        "a": ("200801", "statut"), "c": ("200902", "RA"), "d": ("200801", "sortie")}


@pytest.mark.parametrize("chunksize", [1, 2, 3, 5, 1000])
def test_resultat_independant_de_la_taille_des_blocs(ecrire_perf, chunksize):
    """Un prêt dont les lignes sont réparties sur plusieurs blocs doit donner le même résultat."""
    lignes = [("a", "200803", "03"), ("a", "200801", "00"), ("a", "200802", "01"),
              ("b", "200801", "00"), ("b", "200802", "RA"), ("b", "200801", "00"),
              ("c", "200801", "00", "09"), ("c", "200703", "03"),
              ("d", "200801", "02"), ("d", "200802", "00", "01")]
    p = ecrire_perf(lignes)
    ref = defauts(dates_defaut(p))
    assert ref == {"a": ("200803", "statut"), "b": ("200802", "RA"), "c": ("200703", "statut")}
    assert defauts(dates_defaut(p, chunksize=chunksize)) == ref


def test_forme_du_resultat(ecrire_perf):
    p = ecrire_perf([("a", "200801", "03"), ("a", "200802", "04"), ("b", "200801", "RA")])
    df = dates_defaut(p)
    assert list(df.columns) == ["id", "date_defaut", "declencheur"]
    assert df["id"].is_unique
    assert df["date_defaut"].str.fullmatch(r"\d{6}").all()
    assert set(df["declencheur"]) <= {"statut", "RA", "sortie"}


def test_aucun_defaut_donne_un_tableau_vide_exploitable(ecrire_perf):
    p = ecrire_perf([("a", "200801", "00"), ("a", "200802", "00", "01")])
    df = dates_defaut(p)
    assert len(df) == 0
    assert list(df.columns) == ["id", "date_defaut", "declencheur"]


def test_aucun_defaut_aucun_bloc_ne_contient_de_defaut(ecrire_perf):
    p = ecrire_perf([("a", "200801", "00"), ("b", "200801", "01"), ("c", "200801", "02")])
    assert len(dates_defaut(p, chunksize=1)) == 0


# ------------------------------------------------------------- forbearance (v2)
def marques(df_def, p, **kw):
    out = marquer_forbearance(p, df_def, **kw)
    return dict(zip(out["id"], out["defaut_forbearance"]))


def test_forbearance_avant_pendant_apres_le_defaut(ecrire_perf):
    p = ecrire_perf([
        ("avant", "202003", "00", "", "F"), ("avant", "202006", "03"),
        ("pendant", "202006", "03", "", "F"),
        ("apres", "202006", "03"), ("apres", "202009", "04", "", "F"),
        ("aucun", "202006", "03"),
    ])
    assert marques(dates_defaut(p), p) == {
        "avant": True, "pendant": True, "apres": False, "aucun": False}


def test_sinistre_compte_comme_forbearance(ecrire_perf):
    p = ecrire_perf([("a", "202001", "03", "", "", "Y"), ("b", "202001", "03", "", "", "N")])
    assert marques(dates_defaut(p), p) == {"a": True, "b": False}


@pytest.mark.parametrize("plan", ["R", "T", ""])
def test_seul_le_plan_F_compte(ecrire_perf, plan):
    """Repayment (R) et Trial (T) ne sont pas des forbearances."""
    p = ecrire_perf([("a", "202001", "03", "", plan)])
    assert marques(dates_defaut(p), p) == {"a": False}


def test_un_marqueur_d_un_autre_pret_ne_contamine_pas(ecrire_perf):
    p = ecrire_perf([("a", "202001", "03"), ("b", "202001", "00", "", "F")])
    assert marques(dates_defaut(p), p) == {"a": False}


def test_le_nombre_de_defauts_ne_change_pas_et_l_entree_n_est_pas_modifiee(ecrire_perf):
    p = ecrire_perf([("a", "202001", "03", "", "F"), ("b", "202001", "RA")])
    df = dates_defaut(p)
    avant = df.copy()
    out = marquer_forbearance(p, df)
    pd.testing.assert_frame_equal(df, avant)                # pas d'effet de bord
    assert len(out) == len(df) == 2
    assert out["defaut_forbearance"].dtype == bool


@pytest.mark.parametrize("chunksize", [1, 2, 3, 1000])
def test_forbearance_independante_de_la_taille_des_blocs(ecrire_perf, chunksize):
    p = ecrire_perf([("a", "202001", "00", "", "F"), ("a", "202003", "03"),
                     ("b", "202003", "03"), ("b", "202004", "04", "", "F")])
    assert marques(dates_defaut(p), p, chunksize=chunksize) == {"a": True, "b": False}


def test_forbearance_sans_defaut(ecrire_perf):
    p = ecrire_perf([("a", "202001", "00", "", "F")])
    out = marquer_forbearance(p, dates_defaut(p))
    assert len(out) == 0 and "defaut_forbearance" in out.columns


# ---------------------------------------------------------------- résumé
def test_resume_chiffres_et_affichage(ecrire_perf, capsys):
    p = ecrire_perf([("a", "200801", "03"), ("b", "200901", "RA"), ("c", "200902", "00", "09"),
                     ("d", "200901", "00")])
    n, par_annee = resume_defauts(dates_defaut(p), n_prets=4)
    sortie = capsys.readouterr().out
    assert n == 3
    assert par_annee.to_dict() == {"2008": 1, "2009": 2}
    assert "75.00%" in sortie and "statut" in sortie


def test_resume_sans_defaut(ecrire_perf):
    p = ecrire_perf([("a", "200801", "00")])
    n, par_annee = resume_defauts(dates_defaut(p), n_prets=1)
    assert n == 0 and len(par_annee) == 0
