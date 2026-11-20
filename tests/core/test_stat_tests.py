import numpy as np
import pandas as pd
import pytest
from scipy import stats

from validation_toolkit.core.stat_tests import (
    ajouter_regime, diebold_mariano, moyenne_newey_west)


def test_sans_retard_egale_test_classique():
    x = np.random.default_rng(0).normal(0.5, 1, 30)
    r = moyenne_newey_west(x, retards=0)
    assert r["t"] == pytest.approx(stats.ttest_1samp(x, 0).statistic * np.sqrt(30 / 29))


def test_autocorrelation_augmente_l_erreur_standard():
    rng = np.random.default_rng(1)
    e = rng.normal(size=200)
    x = np.zeros(200)
    for i in range(1, 200):
        x[i] = 0.8 * x[i - 1] + e[i]
    assert moyenne_newey_west(x, retards=10)["se"] > moyenne_newey_west(x, retards=0)["se"]


def test_biais_constant_rejete():
    x = 1.0 + np.random.default_rng(2).normal(0, 0.1, 15)
    assert moyenne_newey_west(x)["p"] < 0.01


def test_trop_court_leve_erreur():
    with pytest.raises(ValueError):
        moyenne_newey_west([1.0, 2.0])


def test_dm_sens():
    ref = np.array([3, 4, 3, 5, 4, 3, 4, 5.0])
    chall = ref - 1 + np.random.default_rng(3).normal(0, 0.05, 8)
    assert diebold_mariano(ref, chall)["moyenne"] > 0


def test_dm_longueurs_differentes():
    with pytest.raises(ValueError):
        diebold_mariano([1, 2, 3], [1, 2])


def test_regime():
    df = pd.DataFrame({"annee": [2019, 2020, 2021]})
    out = ajouter_regime(df, "annee", [("crise", 2020, 2020)])
    assert out["regime"].tolist() == ["normal", "crise", "normal"]
