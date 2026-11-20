"""Contrôles de qualité des données Freddie Mac."""
import pandas as pd


def controle_comptages(orig_csv, perf_csv, id_col, chunksize=500_000):
    """Compte prêts et lignes dans les fichiers origination et performance.

    id_col : nom exact de la colonne identifiant du prêt (identique dans les deux fichiers).
    Lit la performance par blocs. Affiche et retourne un dict de comptages.
    """
    orig = pd.read_csv(orig_csv, dtype=str, keep_default_na=False, usecols=[id_col])
    n_orig_lignes = len(orig)
    ids_orig = set(orig[id_col])

    n_perf_lignes = 0
    ids_perf = set()
    for chunk in pd.read_csv(perf_csv, dtype=str, keep_default_na=False,
                             usecols=[id_col], chunksize=chunksize):
        n_perf_lignes += len(chunk)
        ids_perf.update(chunk[id_col])

    res = {
        "lignes_origination": n_orig_lignes,
        "prets_origination_uniques": len(ids_orig),
        "lignes_performance": n_perf_lignes,
        "prets_performance_uniques": len(ids_perf),
        "lignes_par_pret_moyenne": n_perf_lignes / max(len(ids_perf), 1),
        "prets_orig_sans_perf": len(ids_orig - ids_perf),
        "prets_perf_sans_orig": len(ids_perf - ids_orig),
    }
    for k, v in res.items():
        print(f"{k:32s} {v:,.2f}" if isinstance(v, float) else f"{k:32s} {v:,}")
    return res

def part_fico_manquant(orig_csv, col_fico=0, valeur_manquante="9999"):
    """Part des prêts dont le FICO vaut le code 'non disponible' (9999)."""
    fico = pd.read_csv(orig_csv, dtype=str, keep_default_na=False).iloc[:, col_fico]
    n_manquant = (fico == valeur_manquante).sum()
    print(f"{fico.name} : {n_manquant:,} sur {len(fico):,} non disponibles ({n_manquant / len(fico):.2%})")
    num = pd.to_numeric(fico[fico != valeur_manquante])
    print(f"valeurs disponibles : min {num.min():.0f}, médiane {num.median():.0f}, max {num.max():.0f}")
    return n_manquant

