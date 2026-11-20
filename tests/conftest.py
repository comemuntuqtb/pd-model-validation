"""Outils communs aux tests : fabrication de petits fichiers à réponse connue."""
import pandas as pd
import pytest

COLS_PERF = [
    "Loan Identifier", "Period", "Current Loan Delinquency Status", "Zero Balance Code",
    "Borrower Assistance Plan", "Delinquency Due to Disaster",
]


@pytest.fixture
def ecrire_perf(tmp_path):
    """Retourne f(lignes, nom="perf.csv") qui écrit un CSV de performance.

    Chaque ligne est un tuple (id, période, statut[, code de sortie[, plan[, sinistre]]]);
    les champs omis valent "" (comme dans les vrais fichiers lus en texte).
    """
    def _ecrire(lignes, nom="perf.csv"):
        complet = [tuple(l) + ("",) * (len(COLS_PERF) - len(l)) for l in lignes]
        chemin = tmp_path / nom
        pd.DataFrame(complet, columns=COLS_PERF).to_csv(chemin, index=False)
        return chemin
    return _ecrire
