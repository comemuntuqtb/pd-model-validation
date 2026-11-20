import numpy as np
import pytest

from validation_toolkit.core.discrimination import gini, ks, psi


def test_gini_parfait_et_inverse():
    y = np.array([0, 0, 0, 1, 1])
    assert gini(y, [0.1, 0.2, 0.3, 0.8, 0.9]) == pytest.approx(1)
    assert gini(y, [0.9, 0.8, 0.7, 0.2, 0.1]) == pytest.approx(-1)


def test_gini_aleatoire_proche_de_zero():
    rng = np.random.default_rng(0)
    assert abs(gini(rng.integers(0, 2, 5000), rng.random(5000))) < 0.05


def test_ks_parfait_et_nul():
    y = np.array([0, 0, 1, 1])
    assert ks(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1)
    assert ks(np.array([0, 1, 0, 1]), [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0)


def test_une_seule_classe_leve_erreur():
    with pytest.raises(ValueError):
        gini([0, 0, 0], [0.1, 0.2, 0.3])


def test_longueurs_differentes():
    with pytest.raises(ValueError):
        ks([0, 1], [0.1])


def test_psi_identique_est_nul():
    x = np.random.default_rng(1).normal(size=2000)
    assert psi(x, x) == pytest.approx(0, abs=1e-9)


def test_psi_croit_avec_le_decalage():
    x = np.random.default_rng(2).normal(size=5000)
    assert 0 < psi(x, x + 0.2) < psi(x, x + 1.0)
