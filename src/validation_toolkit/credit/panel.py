"""Panel prêt x année d'observation pour modéliser une PD à 12 mois.

Une ligne = un prêt présent au début de l'année t et pas encore en défaut.
Cible `defaut` = 1 si le prêt fait défaut (règle v2) pendant l'année t, 0 sinon.
Le prêt sort du panel l'année de son défaut ou de sa dernière ligne observée.

Simplifications à documenter :
- la première année est annee_octroi + 1 (première année complète) : les défauts de
  l'année d'octroi n'entrent pas dans le panel ;
- un prêt remboursé en cours d'année compte comme non-défaut pour cette année entière
  (pas de correction de censure) ;
- les caractéristiques sont celles de l'octroi, plus l'âge au 1er janvier.
"""
import numpy as np
import pandas as pd

ID = "Loan Identifier"
NUMERIQUES = {   # colonne du fichier -> (nom court, code "non disponible")
    "Classic FICO®": ("fico", 9999),
    "Original Combined Loan-to-Value (CLTV)": ("cltv", 999),
    "Original Loan-to-Value (LTV)": ("ltv", 999),
    "Original Debt-to-Income (DTI) Ratio": ("dti", 999),
    "Mortgage Insurance Percentage (MI %)": ("mi_pct", 999),
    "Original Interest Rate": ("taux", None),
    "Original UPB": ("upb", None),
    "Original Loan Term": ("duree", None),
    "Number of Units": ("unites", 99),
    "Number of Borrowers": ("emprunteurs", 99),
}
CATEGORIELLES = {
    "Loan Purpose": "objet",
    "Occupancy Status": "occupation",
    "Property Type": "type_bien",
    "First Time Homebuyer Indicator": "primo_accedant",
    "Channel": "canal",
}
PREMIER_PAIEMENT = "First Payment Date"


def derniere_periode(perf_csv, id_col=ID, period_col="Period", chunksize=500_000):
    """Dernière période (YYYYMM, texte) observée pour chaque prêt : Series indexée par prêt."""
    morceaux = []
    for ch in pd.read_csv(perf_csv, dtype=str, keep_default_na=False,
                          usecols=[id_col, period_col], chunksize=chunksize):
        morceaux.append(ch.groupby(id_col)[period_col].max())
    return pd.concat(morceaux).groupby(level=0).max()


def construire_panel(orig_csv, perf_csv, df_def, annee_orig, chunksize=500_000):
    """Construit le panel. df_def : sortie de dates_defaut puis marquer_forbearance
    (colonnes id, date_defaut, defaut_forbearance). annee_orig : millésime (ex. 2007)."""
    if "defaut_forbearance" not in df_def.columns:
        raise KeyError("df_def doit contenir `defaut_forbearance` (appeler marquer_forbearance)")

    colonnes = [ID, PREMIER_PAIEMENT, *NUMERIQUES, *CATEGORIELLES]
    orig = pd.read_csv(orig_csv, dtype=str, keep_default_na=False, usecols=colonnes)

    derniere = derniere_periode(perf_csv, chunksize=chunksize)
    dd = df_def.set_index("id")
    fin_obs = orig[ID].map(derniere).str[:4].astype(int)
    an_def = pd.to_numeric(orig[ID].map(dd["date_defaut"].astype(str).str[:4]),
                           errors="coerce")                               # NaN si jamais en défaut
    debut = annee_orig + 1
    fin = np.minimum(fin_obs, an_def.fillna(np.inf))
    n_annees = (fin - debut + 1).clip(lower=0).astype(int)

    base = pd.DataFrame({"id": orig[ID], "_an_def": an_def})
    base["_forb"] = orig[ID].isin(dd.index[dd["defaut_forbearance"].astype(bool)])
    for col, (nom, special) in NUMERIQUES.items():
        v = pd.to_numeric(orig[col], errors="coerce")
        base[nom] = v.mask(v == special) if special is not None else v
    for col, nom in CATEGORIELLES.items():
        base[nom] = orig[col]
    fp = orig[PREMIER_PAIEMENT]
    base["_fp_annee"] = fp.str[:4].astype(int)
    base["_fp_mois"] = fp.str[4:6].astype(int)

    panel = base.loc[base.index.repeat(n_annees)].copy()
    panel["annee"] = debut + panel.groupby("id").cumcount()
    panel["defaut"] = (panel["annee"] == panel["_an_def"]).astype(int)
    panel["defaut_hors_forbearance"] = (panel["defaut"].eq(1) & ~panel["_forb"]).astype(int)
    panel["age_mois"] = ((panel["annee"] - panel["_fp_annee"]) * 12 - (panel["_fp_mois"] - 1)).clip(lower=0)
    panel = panel.drop(columns=["_an_def", "_forb", "_fp_annee", "_fp_mois"])
    return panel.reset_index(drop=True)


def resume_panel(panel):
    """Lignes, défauts et taux par année d'observation."""
    r = panel.groupby("annee").agg(prets=("id", "size"), defauts=("defaut", "sum"),
                                   defauts_hors_forbearance=("defaut_hors_forbearance", "sum"))
    r["taux"] = r["defauts"] / r["prets"]
    print(r.round(4))
    return r
