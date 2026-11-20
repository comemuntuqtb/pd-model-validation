"""Tests de calibration d'une probabilité prédite contre un taux observé.
cote="sous" : H1 = le modèle SOUS-estime le risque (p-value = P(X >= observé)).
cote="sur"  : H1 = le modèle SUR-estime le risque (p-value = P(X <= observé))."""
import numpy as np
from scipy import stats


def _verifier_cote(cote):
    if cote not in ("sous", "sur"):
        raise ValueError("cote doit valoir 'sous' ou 'sur'")


def binomial_exact(n, defauts, pd_predite, cote="sous"):
    """Défauts supposés indépendants, de probabilité pd_predite."""
    _verifier_cote(cote)
    if not 0 < pd_predite < 1:
        raise ValueError("pd_predite doit être strictement entre 0 et 1")
    if cote == "sous":
        return float(stats.binom.sf(defauts - 1, n, pd_predite))
    return float(stats.binom.cdf(defauts, n, pd_predite))


def vasicek(n, defauts, pd_predite, rho, cote="sous"):
    """Test corrigé de la corrélation d'actifs (modèle à un facteur de Vasicek, grand portefeuille).
    Sous H0, le taux de défaut X vérifie P(X <= x) = Φ((√(1−ρ)·Φ⁻¹(x) − Φ⁻¹(PD)) / √ρ).
    `n` n'intervient que par le taux observé : l'approximation suppose n grand."""
    _verifier_cote(cote)
    if not 0 < pd_predite < 1:
        raise ValueError("pd_predite doit être strictement entre 0 et 1")
    if not 0 < rho < 1:
        raise ValueError("rho doit être strictement entre 0 et 1")
    x = defauts / n
    if x <= 0:
        return 1.0 if cote == "sous" else 0.0
    if x >= 1:
        return 0.0 if cote == "sous" else 1.0
    cdf = stats.norm.cdf((np.sqrt(1 - rho) * stats.norm.ppf(x) - stats.norm.ppf(pd_predite)) / np.sqrt(rho))
    return float(1 - cdf) if cote == "sous" else float(cdf)


def intervalle_exact(n, defauts, niveau=0.95):
    """Intervalle de confiance exact (Clopper-Pearson) de la probabilité de défaut. Valable même à 0 défaut."""
    if n <= 0 or not 0 <= defauts <= n:
        raise ValueError("il faut n > 0 et 0 <= defauts <= n")
    a = 1 - niveau
    bas = 0.0 if defauts == 0 else float(stats.beta.ppf(a / 2, defauts, n - defauts + 1))
    haut = 1.0 if defauts == n else float(stats.beta.ppf(1 - a / 2, defauts + 1, n - defauts))
    return bas, haut


def puissance_binomial(n, pd0, facteur, alpha=0.05):
    """Probabilité que le test binomial exact unilatéral (H1 : sous-estimation) rejette H0 : PD = pd0
    quand la vraie PD vaut facteur × pd0. Test exact : sa taille réelle est ≤ alpha (loi discrète)."""
    if not 0 < pd0 < 1 or not 0 < facteur * pd0 < 1:
        raise ValueError("pd0 et facteur * pd0 doivent être strictement entre 0 et 1")
    k = stats.binom.isf(alpha, n, pd0)      # on rejette si défauts > k
    return float(stats.binom.sf(k, n, facteur * pd0))

