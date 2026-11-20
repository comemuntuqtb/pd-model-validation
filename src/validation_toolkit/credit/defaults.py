"""Règle de défaut Freddie Mac (v2 : v1 + marquage forbearance/sinistre)."""
import pandas as pd

ZB_DEFAUT = {"02", "03", "09", "15"}
SEUIL_STATUT = 3  # >= 03 : 90 jours d'impayés ou plus


def _flags(chunk, c_status, c_zb):
    """Retourne trois booléens par ligne : statut >= 03, statut RA, code de sortie de défaut."""
    st = chunk[c_status]
    num = pd.to_numeric(st.where(~st.isin(["RA", "XX", ""])), errors="coerce")
    f_statut = num >= SEUIL_STATUT
    f_ra = st == "RA"
    f_zb = chunk[c_zb].isin(ZB_DEFAUT)
    return f_statut, f_ra, f_zb


def dates_defaut(perf_csv, id_col="Loan Identifier", period_col="Period",
                 status_col="Current Loan Delinquency Status",
                 zb_col="Zero Balance Code", chunksize=500_000):
    """Règle de défaut v1. Retourne un DataFrame, une ligne par prêt en défaut :
    id, date de défaut (YYYYMM, premier mois où une condition est vraie)
    et déclencheur (statut / RA / sortie) de ce premier mois.
    """
    usecols = [id_col, period_col, status_col, zb_col]
    morceaux = []
    for chunk in pd.read_csv(perf_csv, dtype=str, keep_default_na=False,
                             usecols=usecols, chunksize=chunksize):
        f_st, f_ra, f_zb = _flags(chunk, status_col, zb_col)
        mask = f_st | f_ra | f_zb
        sub = chunk.loc[mask, [id_col, period_col]].copy()
        sub["declencheur"] = "statut"
        sub.loc[f_ra[mask], "declencheur"] = "RA"
        sub.loc[f_zb[mask] & ~f_st[mask] & ~f_ra[mask], "declencheur"] = "sortie"
        morceaux.append(sub)
    tout = pd.concat(morceaux, ignore_index=True)
    tout = tout.sort_values([id_col, period_col])  # YYYYMM texte : l'ordre alphabétique = chronologique
    premier = tout.drop_duplicates(id_col, keep="first").reset_index(drop=True)
    return premier.rename(columns={id_col: "id", period_col: "date_defaut"})


def marquer_forbearance(perf_csv, df_def, id_col="Loan Identifier", period_col="Period",
                        plan_col="Borrower Assistance Plan",
                        disaster_col="Delinquency Due to Disaster", chunksize=500_000):
    """Règle de défaut v2 : ajoute à df_def la colonne booléenne `defaut_forbearance`,
    vraie si le prêt a un marqueur (plan = F ou sinistre = Y) sur une ligne à la date
    de défaut ou avant. Les défauts restent comptés ; la colonne permet de calculer
    les taux avec et sans ces prêts.
    """
    if df_def.empty:
        out = df_def.copy()
        out["defaut_forbearance"] = pd.Series(dtype=bool)
        return out
    date = df_def.set_index("id")["date_defaut"]
    marques = set()
    for chunk in pd.read_csv(perf_csv, dtype=str, keep_default_na=False,
                             usecols=[id_col, period_col, plan_col, disaster_col],
                             chunksize=chunksize):
        chunk = chunk[chunk[id_col].isin(date.index)]
        chunk = chunk[chunk[period_col] <= chunk[id_col].map(date)]
        mark = (chunk[plan_col] == "F") | (chunk[disaster_col] == "Y")
        marques.update(chunk.loc[mark, id_col])
    out = df_def.copy()
    out["defaut_forbearance"] = out["id"].isin(marques)
    return out



def resume_defauts(df_def, n_prets, annee_orig=None):
    """Affiche et retourne le nombre de défauts, le taux, la répartition par déclencheur et par année."""
    n = len(df_def)
    print(f"défauts : {n:,} sur {n_prets:,} prêts, taux = {n / n_prets:.2%}")
    print("\npar déclencheur du premier mois :")
    print(df_def["declencheur"].value_counts())
    print("\npar année calendaire de défaut :")
    par_annee = df_def["date_defaut"].str[:4].value_counts().sort_index()
    print(par_annee)
    return n, par_annee




COL_REGLEMENT = "Underwriting Defect and Major Servicing Defect Settlement Date"


def sortie_des_defauts(perf_csv, df_def, id_col="Loan Identifier", period_col="Period",
                       zb_col="Zero Balance Code", loss_col="Actual Loss", chunksize=500_000):
    """Pour chaque prêt en défaut, regarde sa DERNIÈRE ligne (dernier mois observé) et
    classe sa sortie. Retourne un DataFrame (une ligne par prêt en défaut) avec la colonne
    `sortie`, et affiche la répartition.
    """
    ids = set(df_def["id"])
    morceaux = []
    for ch in pd.read_csv(perf_csv, dtype=str, keep_default_na=False,
                          usecols=[id_col, period_col, zb_col, loss_col], chunksize=chunksize):
        ch = ch[ch[id_col].isin(ids)].sort_values([id_col, period_col])
        morceaux.append(ch.drop_duplicates(id_col, keep="last"))   # dernière ligne du prêt dans ce bloc
    tout = pd.concat(morceaux).sort_values([id_col, period_col])
    derniere = tout.drop_duplicates(id_col, keep="last")           # dernière ligne sur tous les blocs

    def classer(r):
        zb, perte = r[zb_col], r[loss_col] != ""
        if zb == "":
            return "encore en vie (pas de sortie)"
        if zb in ("02", "03", "09", "15"):
            return "sortie 02/03/09/15 avec perte" if perte else "sortie 02/03/09/15 SANS perte"
        return f"sortie {zb}"

    out = derniere.assign(sortie=derniere.apply(classer, axis=1))
    print(out["sortie"].value_counts())
    print(f"\ntotal : {len(out):,} défauts")
    return out

def marquer_defaut_souscription(perf_csv, df_def, id_col="Loan Identifier",
                                col=COL_REGLEMENT, chunksize=500_000):
    """Ajoute à df_def la colonne booléenne `defaut_souscription` : vraie si le prêt a une
    date de règlement de défaut de souscription / de gestion sur au moins une ligne.
    Les défauts restent comptés pour la PD ; la colonne sert à exclure ces prêts du LGD.
    """
    ids = set(df_def["id"])
    marques = set()
    for ch in pd.read_csv(perf_csv, dtype=str, keep_default_na=False,
                          usecols=[id_col, col], chunksize=chunksize):
        ch = ch[ch[id_col].isin(ids)]
        marques |= set(ch.loc[ch[col] != "", id_col])
    out = df_def.copy()
    out["defaut_souscription"] = out["id"].isin(marques)
    return out
