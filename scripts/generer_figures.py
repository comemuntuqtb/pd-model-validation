"""Génère les figures du README (docs/images/*.png) à partir de data/processed.
Ne dessine que des agrégats : aucune ligne de prêt n'est affichée. Lancer depuis la racine : python scripts/generer_figures.py"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, FixedLocator

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
from validation_toolkit.core.calibration_tests import puissance_binomial
from validation_toolkit.credit.lgd import contraste_regimes, lgd_par_cohorte

PROC, SORTIE = RACINE / "data" / "processed", RACINE / "docs" / "images"
SORTIE.mkdir(parents=True, exist_ok=True)

# palette de référence validée (bleu / orange) ; l'observé est en encre neutre
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
BLEU, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def fr(v, d=2):
    return f"{v:.{d}f}".replace(".", ",")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def pct(ax, axe="y"):
    f = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",") + " %")
    (ax.yaxis if axe == "y" else ax.xaxis).set_major_formatter(f)


def titre(fig, haut, bas):
    fig.text(0.012, 0.975, haut, ha="left", va="top", fontsize=12.5, fontweight="semibold")
    fig.text(0.012, 0.915, bas, ha="left", va="top", fontsize=9.5, color=INK2)


def enregistrer(fig, nom):
    fig.savefig(SORTIE / nom, dpi=170)
    plt.close(fig)
    print("écrit :", SORTIE / nom)


def annuel(an, cible):
    c = pd.read_csv(PROC / f"tableau_coeur_{an}_{cible}.csv")
    col = "observe_hors_forbearance" if cible == "principal" else "observe"
    out = {}
    for m, g in c.groupby("modele"):
        h = g.groupby("date")
        out[m] = pd.DataFrame({
            "obs": h.apply(lambda x: np.average(x[col], weights=x["n"]), include_groups=False),
            "pred": h.apply(lambda x: np.average(x["predit"], weights=x["n"]), include_groups=False)})
    return out


def courbes(ax, d):
    kw = dict(lw=2, marker="o", ms=6, mec=SURF, mew=1.5)
    ax.plot(d["ref"].index, 100 * d["ref"]["obs"], color=INK, label="observé", **kw)
    ax.plot(d["ref"].index, 100 * d["ref"]["pred"], color=BLEU, label="logistique (référence)", **kw)
    ax.plot(d["chall"].index, 100 * d["chall"]["pred"], color=ORANGE, label="boosting (challenger)", **kw)


def legende(fig, ax, y=0.01):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, y))


# ---- 1. contamination par la forbearance (millésime 2015)
def fig_forbearance():
    sens, princ = annuel(2015, "sensibilite"), annuel(2015, "principal")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    fig.subplots_adjust(top=0.74, bottom=0.2, left=0.07, right=0.985, wspace=0.08)
    for ax, d, t in ((axes[0], sens, "Entraîné avec la forbearance (cible : défaut)"),
                     (axes[1], princ, "Entraîné hors forbearance (cible principale)")):
        courbes(ax, d); style(ax); pct(ax)
        ax.set_title(t, loc="left", fontsize=10.5, color=INK)
    a0 = axes[0]
    an_max = sens["ref"]["pred"].idxmax()
    a0.annotate(f"{fr(100 * sens['ref']['pred'].max())} %", (an_max, 100 * sens["ref"]["pred"].max()),
                xytext=(7, 3), textcoords="offset points", fontsize=9, color=INK2)
    a0.annotate(f"observé {fr(100 * sens['ref'].loc[2020, 'obs'])} %", (2020, 100 * sens["ref"].loc[2020, "obs"]),
                xytext=(0, 9), textcoords="offset points", fontsize=9, color=INK2, ha="center")
    axes[0].set_ylim(0, 5.6)
    titre(fig, "La forbearance de 2020 fausse l'entraînement, pas seulement le test",
          "Millésime 2015, PD annuelle moyenne prédite contre taux de défaut observé, 2019–2026")
    legende(fig, axes[0])
    enregistrer(fig, "01_forbearance_2015.png")


# ---- 2. millésime 2007 : observé contre prédit
def fig_2007():
    d = annuel(2007, "principal")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(top=0.76, bottom=0.2, left=0.08, right=0.97)
    courbes(ax, d); style(ax); pct(ax)
    ax.set_ylim(0, 10.5); ax.set_xlabel("année de test")
    a = d["ref"].index[0]
    ax.text(0.985, 0.95, f"en {a} : {fr(100 * d['ref'].loc[a, 'pred'])} % prédit par la logistique\n"
            f"contre {fr(100 * d['ref'].loc[a, 'obs'])} % observé", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=INK2)
    titre(fig, "Millésime 2007 : la logistique surestime le risque, le boosting suit l'observé",
          "Walk-forward, cible principale (hors forbearance), PD annuelle moyenne, 2011–2026")
    legende(fig, ax)
    enregistrer(fig, "02_millesime_2007.png")


# ---- 3. calibration par décile
def fig_calibration():
    c = pd.read_csv(PROC / "tableau_coeur_2007_principal.csv")
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    fig.subplots_adjust(top=0.8, bottom=0.17, left=0.13, right=0.96)
    lim = (0.2, 30)
    ax.plot(lim, lim, color=INK2, lw=1, zorder=1)
    ax.text(0.27, 0.36, "calibration parfaite", color=INK2, fontsize=9, rotation=34, ha="left", va="bottom")
    for m, couleur, nom in (("ref", BLEU, "logistique (référence)"), ("chall", ORANGE, "boosting (challenger)")):
        g = c[c["modele"] == m].groupby("groupe")
        pred = 100 * g.apply(lambda x: np.average(x["predit"], weights=x["n"]), include_groups=False)
        obs = 100 * g["n_evenements_hors_forbearance"].sum() / g["n"].sum()
        ax.plot(pred, obs, "o", color=couleur, ms=9, mec=SURF, mew=2, label=nom, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
    ticks = [0.3, 1, 3, 10, 30]
    for axe in (ax.xaxis, ax.yaxis):
        axe.set_major_locator(FixedLocator(ticks)); axe.set_minor_locator(FixedLocator([]))
    pct(ax); pct(ax, "x")
    ax.set_xlabel("PD prédite (moyenne par classe, 2011–2026)"); ax.set_ylabel("taux de défaut observé")
    style(ax); ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="upper left")
    titre(fig, "Calibration par décile, millésime 2007",
          "Le boosting colle à la diagonale ; la logistique est dessous (modèle prudent). Axes log")
    enregistrer(fig, "03_calibration_deciles_2007.png")


# ---- 4. LGD par année de défaut
def fig_lgd():
    lgd = pd.read_pickle(PROC / "lgd_2007.pkl")
    coh = lgd_par_cohorte(lgd, seuil=30)
    ok = coh[coh["concluante"]]
    regimes = [("crise_2008", 2008, 2010), ("stress_2020", 2020, 2020)]
    r = contraste_regimes(lgd, regimes, "crise_2008", nb_tirages=2000)
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    fig.subplots_adjust(top=0.78, bottom=0.1, left=0.09, right=0.97, hspace=0.12)
    for ax in (a, b):
        ax.axvspan(2007.5, 2010.5, color="#ececea", zorder=0)
    a.plot(ok["annee_defaut"], 100 * ok["lgd_ponderee"], color=BLEU, lw=2, marker="o", ms=6, mec=SURF, mew=1.5)
    a.set_ylim(30, 62); style(a); pct(a); a.set_ylabel("LGD pondérée")
    a.text(2009, 60.5, "crise 2008–2010", ha="center", fontsize=9, color=INK2)
    a.text(0.985, 0.95, f"écart crise − autres années : +{fr(100 * r['ecart'], 1)} points\n"
           f"intervalle bootstrap {fr(100 * r['ic95'][0], 1)} à {fr(100 * r['ic95'][1], 1)} (trop étroit)",
           transform=a.transAxes, ha="right", va="top", fontsize=9, color=INK2)
    b.bar(coh["annee_defaut"], coh["n"], width=0.6, color="#a9a7a0")
    style(b); b.set_ylabel("prêts"); b.set_xlabel("année du défaut")
    b.set_xticks(range(2007, 2019, 2)); b.set_xlim(2006.4, 2018.6)
    titre(fig, "Millésime 2007 : la LGD des défauts de crise est plus élevée de 8 points",
          "LGD réalisée = perte / UPB au défaut, cohortes d'au moins 30 prêts ; pas d'actualisation, défauts liquidés seulement")
    enregistrer(fig, "04_lgd_2007.png")


# ---- 5. puissance du test binomial selon le nombre de défauts
def fig_puissance():
    p0 = 0.000479   # PD moyenne prédite du segment à faible taux de défaut (notebook 07)
    D = np.unique(np.concatenate([np.geomspace(1, 100, 120), [5, 10, 20, 30]]))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(top=0.76, bottom=0.2, left=0.08, right=0.97)
    for f, couleur, nom in ((2, BLEU, "PD réelle 2 fois l'annoncée"), (3, ORANGE, "PD réelle 3 fois l'annoncée")):
        p = [100 * puissance_binomial(max(1, int(round(d / p0))), p0, f) for d in D]
        ax.plot(D, p, color=couleur, lw=2, label=nom)
    for d in (5, 10):
        v = 100 * puissance_binomial(int(round(d / p0)), p0, 2)
        ax.plot([d], [v], "o", color=BLEU, ms=8, mec=SURF, mew=2)
        ax.annotate(f"{d} défauts : {v:.0f} %", (d, v), xytext=(8, -14), textcoords="offset points", fontsize=9, color=INK2)
    ax.axvline(30, color=INK2, lw=1)
    ax.text(31, 6, "seuil du projet :\n30 défauts", fontsize=9, color=INK2, va="bottom")
    ax.text(1.05, 8, "dents de scie : le test exact est discret", fontsize=8.5, color=INK2)
    ax.axhline(80, color=GRID, lw=1.2)
    ax.text(1.05, 81.5, "80 %", fontsize=9, color=INK2)
    ax.set_xscale("log"); ax.set_ylim(0, 103)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 30, 50, 100]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    pct(ax); style(ax)
    ax.set_xlabel("nombre de défauts attendus (échelle logarithmique)"); ax.set_ylabel("puissance du test")
    titre(fig, "Sous 10 défauts attendus, le test binomial rate souvent une PD deux fois trop faible",
          "Probabilité de rejeter à 5 % la PD annoncée (0,048 %) quand la vraie PD est plus élevée ; défauts supposés indépendants")
    legende(fig, ax)
    enregistrer(fig, "05_puissance_test_binomial.png")


if __name__ == "__main__":
    fig_forbearance(); fig_2007(); fig_calibration(); fig_lgd(); fig_puissance()
