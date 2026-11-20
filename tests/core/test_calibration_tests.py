import pytest
from scipy import stats

from validation_toolkit.core.calibration_tests import binomial_exact, vasicek


def test_binomial_egale_scipy():
    assert binomial_exact(1000, 30, 0.02) == pytest.approx(stats.binomtest(30, 1000, 0.02, alternative="greater").pvalue)


def test_vasicek_moins_severe_que_binomial():
    # même données : avec corrélation positive, l'écart est moins surprenant
    assert vasicek(10000, 260, 0.02, rho=0.15) > binomial_exact(10000, 260, 0.02)


def test_vasicek_decroit_avec_les_defauts():
    p = [vasicek(10000, d, 0.02, 0.15) for d in (150, 200, 300, 500)]
    assert p == sorted(p, reverse=True)


def test_vasicek_taux_egal_a_la_pd_donne_une_p_value_moyenne():
    assert 0.3 < vasicek(100000, 2000, 0.02, 0.15) < 0.7


def test_vasicek_zero_defaut():
    assert vasicek(1000, 0, 0.02, 0.15) == 1.0


@pytest.mark.parametrize("pd_, rho", [(0, 0.1), (1, 0.1), (0.02, 0), (0.02, 1)])
def test_parametres_invalides(pd_, rho):
    with pytest.raises(ValueError):
        vasicek(1000, 20, pd_, rho)


def test_cotes_complementaires_vasicek():
    assert vasicek(10000, 200, 0.02, 0.15, "sous") + vasicek(10000, 200, 0.02, 0.15, "sur") == pytest.approx(1)


def test_cote_sur_binomial():
    assert binomial_exact(1000, 5, 0.02, "sur") == pytest.approx(stats.binom.cdf(5, 1000, 0.02))
    assert binomial_exact(1000, 5, 0.02, "sur") < 0.01


def test_cote_invalide():
    with pytest.raises(ValueError):
        vasicek(1000, 20, 0.02, 0.15, "milieu")


from validation_toolkit.core.calibration_tests import intervalle_exact, puissance_binomial


def test_intervalle_zero_defaut_connu():
    bas, haut = intervalle_exact(10, 0)
    assert bas == 0.0 and haut == pytest.approx(1 - 0.025 ** (1 / 10))


def test_intervalle_contient_le_taux_observe_et_se_resserre():
    b1, h1 = intervalle_exact(100, 5)
    b2, h2 = intervalle_exact(10000, 500)
    assert b1 < 0.05 < h1 and b2 < 0.05 < h2 and (h2 - b2) < (h1 - b1)


def test_intervalle_egal_scipy():
    bas, haut = intervalle_exact(200, 7)
    ref = stats.binomtest(7, 200).proportion_ci(method="exact")
    assert (bas, haut) == pytest.approx((ref.low, ref.high))


def test_intervalle_invalide():
    with pytest.raises(ValueError):
        intervalle_exact(10, 11)


def test_puissance_croit_avec_n_et_le_facteur():
    assert puissance_binomial(20000, 0.002, 2) > puissance_binomial(2000, 0.002, 2)
    assert puissance_binomial(5000, 0.002, 3) > puissance_binomial(5000, 0.002, 1.5)


def test_puissance_sans_ecart_ne_depasse_pas_alpha():
    assert puissance_binomial(5000, 0.002, 1.0) <= 0.05
