# LGD réalisée : définition retenue et limites

Version du 2026-11-20. Code : `src/validation_toolkit/credit/lgd.py`. Tests : `tests/credit/test_lgd.py`.

## Ce qui est calculé (définition « a » : perte en cas de liquidation)

Pour chaque prêt en défaut sorti par un code 02, 03, 09 ou 15, avec une perte renseignée et sans règlement de défaut de souscription :

    LGD = Actual Loss / Current Actual UPB au mois du défaut

- `Actual Loss` (guide Freddie Mac, p. 13) = (UPB retiré + produit net de vente) + intérêts impayés + frais + recouvrements d'assurance + autres recouvrements. Perte positive, gain négatif. Elle est donc **après assurance hypothécaire**.
- Cohorte = **année du défaut**, pas année de la vente : la perte arrive des années après le défaut.
- LGD d'une cohorte = Σ pertes / Σ UPB au défaut (pondérée par l'exposition). La moyenne simple et la médiane sont données à côté.
- Contrôle : un second dénominateur, l'UPB retiré à la sortie, donne la même LGD pour la quasi-totalité des prêts (écart médian nul).
- Les valeurs hors de [0 ; 1] sont conservées : une perte négative est un gain réel, une LGD supérieure à 1 vient des intérêts, frais et impôts qui s'ajoutent à l'UPB.

## Pourquoi la définition « b » (LGD « workout », avec les défauts guéris) n'est pas faite

La LGD « b » rapporte la perte à **tous** les défauts, y compris ceux qui n'ont pas coûté d'argent (prêt remboursé ou redevenu sain). C'est la notion la plus proche d'une LGD réglementaire, à ma connaissance, mais cela reste à vérifier dans les orientations de l'EBA sur l'estimation de la LGD. Elle demande des décisions que les données ne tranchent pas :

1. **Quand un défaut est-il guéri ?** Il faut une règle (nombre de mois sans retard, etc.). Le choix change le résultat.
2. **Les défauts non résolus.** Un prêt en défaut dont on ne connaît pas encore l'issue n'a pas de perte finale. Les ignorer biaise la LGD vers les issues rapides ; les compter à zéro la sous-estime.
3. **Les flux ne sont pas datés.** Freddie Mac donne une perte nette à la liquidation, pas le détail des encaissements mois par mois.
4. **L'actualisation.** Une LGD économique actualise les flux au taux approprié. Ce taux est lui aussi un choix à justifier.

Une règle de guérison, une règle de censure et un taux d'actualisation à fixer avant de regarder les résultats représentent un travail de définition plus long que le calcul lui-même. Il est hors périmètre de ce projet.

## Limites de la définition « a » à citer avec les résultats

- **Elle surestime la perte par défaut.** Elle ne compte que les défauts qui finissent en liquidation et ignore ceux qui s'en sortent sans perte.
- **Pas d'actualisation.** Les pertes sont nominales, alors qu'une LGD économique est actualisée. Un délai médian de 14 à 28 mois entre le défaut et la sortie rend l'écart non négligeable.
- **Les coûts de modification sont exclus** de `Actual Loss` (guide, p. 13) : la perte des prêts modifiés est sous-estimée.
- **Sélection des cohortes récentes.** Les défauts récents n'apparaissent que s'ils ont été résolus vite, et la perte est vide pour les sorties des 3 derniers mois avant la coupure des données. Le délai médian baisse de 27 mois (cohorte 2008) à 14 mois (2012), ce qui montre l'effet.
- **Un seul millésime exploitable.** 2007 donne 4 043 prêts ; 2015 en donne 134, dont une seule année de défaut dépasse 30 prêts. Aucune conclusion statistique sur 2015.
- **L'année du défaut et l'âge du prêt sont liés** à l'intérieur d'un millésime : on ne peut pas séparer l'effet « crise » de l'effet « prêt plus ancien ».
- **Contraste crise / normal.** Il est descriptif, et pas l'ajustement downturn de l'EBA/GL/2019/03, qu'il faudrait lire avant de l'implémenter. L'intervalle de confiance est un bootstrap sur les prêts : il ne tient pas compte de la corrélation des pertes entre prêts d'une même année (prix de l'immobilier). Il est donc trop étroit pour la question « la crise a-t-elle une LGD plus élevée ? ».
