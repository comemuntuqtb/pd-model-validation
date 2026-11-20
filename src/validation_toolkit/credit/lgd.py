"""LGD réalisée (définition a : perte en cas de liquidation) sur les défauts sortis par 02/03/09/15."""
import numpy as np
import pandas as pd

from validation_toolkit.core.stat_tests import ajouter_regime


def echantillon_lgd(sortie_df, df_def, id_col="Loan Identifier"):
    """Prêts utilisables pour le LGD : sortie 02/03/09/15 avec perte renseignée ET sans
    règlement de défaut de souscription. Affiche les effectifs et retourne la liste d'identifiants."""
    m = sortie_df[[id_col, "sortie"]].merge(df_def[["id", "defaut_souscription"]],
                                            left_on=id_col, right_on="id")
    avec_perte = m["sortie"] == "sortie 02/03/09/15 avec perte"
    garde = avec_perte & ~m["defaut_souscription"]
    print(f"défauts : {len(m):,} | avec perte renseignée : {avec_perte.sum():,} | "
          f"dont retirés (règlement de défaut de souscription) : {(avec_perte & m['defaut_souscription']).sum():,} | "
          f"échantillon LGD : {garde.sum():,}")
    return m.loc[garde, id_col].tolist()


def lgd_par_pret(perf_csv, ids, df_def, id_col="Loan Identifier", period_col="Period",
                 upb_col="Current Actual UPB", upb_sortie_col="Zero Balance Removal UPB",
                 perte_col="Actual Loss", sortie_col="Zero Balance Effective Date", chunksize=500_000):
    """LGD de chaque prêt de l'échantillon : perte réelle / UPB au mois du défaut.

    Retourne un DataFrame (une ligne par prêt) : id, date_defaut, annee_defaut, perte, upb_defaut,
    upb_sortie, lgd (dénominateur = UPB au mois du défaut), lgd_upb_sortie (dénominateur = UPB retiré
    à la sortie, pour contrôle), delai_mois (entre le mois du défaut et la sortie).
    La perte est celle de la dernière ligne du prêt. Une perte négative (gain) est conservée.
    """
    ids = set(ids)
    date = df_def.set_index("id")["date_defaut"]
    date = date[date.index.isin(ids)]
    au_defaut, dernieres = [], []
    cols = [id_col, period_col, upb_col, upb_sortie_col, perte_col, sortie_col]
    for ch in pd.read_csv(perf_csv, dtype=str, keep_default_na=False, usecols=cols, chunksize=chunksize):
        ch = ch[ch[id_col].isin(ids)]
        if ch.empty:
            continue
        au_defaut.append(ch[ch[period_col] == ch[id_col].map(date)][[id_col, upb_col]])
        dernieres.append(ch.sort_values([id_col, period_col]).drop_duplicates(id_col, keep="last"))
    if not dernieres:
        raise ValueError("aucun prêt de l'échantillon trouvé dans le fichier de performance")
    fin = pd.concat(dernieres).sort_values([id_col, period_col]).drop_duplicates(id_col, keep="last")
    deb = pd.concat(au_defaut).drop_duplicates(id_col, keep="last") if au_defaut else \
        pd.DataFrame(columns=[id_col, upb_col])

    out = pd.DataFrame({"id": fin[id_col].values})
    out["date_defaut"] = out["id"].map(date)
    out["annee_defaut"] = pd.to_numeric(out["date_defaut"].str[:4], errors="coerce")
    out["perte"] = pd.to_numeric(fin[perte_col], errors="coerce").values
    out["upb_sortie"] = pd.to_numeric(fin[upb_sortie_col], errors="coerce").values
    out["upb_defaut"] = pd.to_numeric(out["id"].map(deb.set_index(id_col)[upb_col]), errors="coerce")
    out.loc[out["upb_defaut"] <= 0, "upb_defaut"] = np.nan
    out.loc[out["upb_sortie"] <= 0, "upb_sortie"] = np.nan
    out["lgd"] = out["perte"] / out["upb_defaut"]
    out["lgd_upb_sortie"] = out["perte"] / out["upb_sortie"]
    d, f = pd.to_datetime(out["date_defaut"].values, format="%Y%m", errors="coerce"), \
        pd.to_datetime(fin[sortie_col].values, format="%Y%m", errors="coerce")
    out["delai_mois"] = ((f.year - d.year) * 12 + (f.month - d.month)).to_numpy()
    return out


def _ratio(perte, upb):
    return perte.sum() / upb.sum()


def lgd_par_cohorte(df_lgd, seuil=30):
    """Par année de défaut : effectif, LGD pondérée par l'exposition (Σ pertes / Σ UPB au défaut),
    LGD moyenne simple, médiane. `concluante` = au moins `seuil` prêts. Exclut les prêts sans UPB au défaut."""
    d = df_lgd.dropna(subset=["lgd", "upb_defaut"])
    lignes = []
    for a, g in d.groupby("annee_defaut"):
        lignes.append({"annee_defaut": int(a), "n": len(g), "lgd_ponderee": _ratio(g["perte"], g["upb_defaut"]),
                       "lgd_moyenne": g["lgd"].mean(), "lgd_mediane": g["lgd"].median(),
                       "delai_median_mois": g["delai_mois"].median(), "concluante": len(g) >= seuil})
    return pd.DataFrame(lignes)


def contraste_regimes(df_lgd, regimes, regime_a, regime_b="normal", nb_tirages=2000, graine=0):
    """Écart de LGD pondérée (regime_a − regime_b) avec intervalle de confiance à 95 % par bootstrap
    (tirage des prêts avec remise, dans chaque régime). Les régimes sont attribués à l'année du défaut."""
    d = ajouter_regime(df_lgd.dropna(subset=["lgd", "upb_defaut"]), "annee_defaut", regimes)
    ga, gb = d[d["regime"] == regime_a], d[d["regime"] == regime_b]
    if len(ga) < 2 or len(gb) < 2:
        raise ValueError("au moins 2 prêts nécessaires dans chaque régime")
    rng = np.random.default_rng(graine)

    def tirages(g):
        pe, up = g["perte"].to_numpy(), g["upb_defaut"].to_numpy()
        idx = rng.integers(0, len(g), size=(nb_tirages, len(g)))
        return pe[idx].sum(axis=1) / up[idx].sum(axis=1)

    ecarts = tirages(ga) - tirages(gb)
    la, lb = _ratio(ga["perte"], ga["upb_defaut"]), _ratio(gb["perte"], gb["upb_defaut"])
    return {"n_a": len(ga), "n_b": len(gb), "lgd_a": la, "lgd_b": lb, "ecart": la - lb,
            "ic95": (float(np.quantile(ecarts, 0.025)), float(np.quantile(ecarts, 0.975)))}

