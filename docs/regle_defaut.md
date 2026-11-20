# Règle de défaut appliquée aux données Freddie Mac

Version du 2026-11-20. Code : `src/validation_toolkit/credit/defaults.py`. Paramètres figés dans `config/freddie_mac.yaml`.

## Ce que dit (et ne dit pas) la définition réglementaire

La définition prudentielle d'un défaut retient deux critères : un retard de paiement de plus de 90 jours, ou l'improbabilité que l'emprunteur paie (art. 178 du règlement CRR). **Je l'ai citée de mémoire, sans relire le texte.** Les orientations EBA/GL/2016/07 sur l'application de cette définition sont à lire avant de prétendre que la règle ci-dessous les respecte ; je ne les ai pas lues.

Les seuils de matérialité (montant minimal d'impayé) ne sont pas appliqués : le fichier ne donne aucun montant d'impayé. Ranger un bien saisi (RA) dans « improbabilité de paiement » est un choix de modélisation.

## Règle v1

Un prêt est en défaut **dès le premier mois** où l'une de ces conditions est vraie :

1. Le statut d'impayés (`Current Loan Delinquency Status`) est numérique et vaut **03 ou plus** (90 jours d'impayés ou plus). La valeur 99 est un **plafond** de l'indicateur, pas « non disponible ».
2. Le statut vaut **RA** (bien saisi par le prêteur, REO).
3. Le `Zero Balance Code` est **02** (vente à un tiers), **03** (vente à perte ou passage en perte), **09** (cession du bien saisi) ou **15** (vente de prêts).

Exclusions :
- Code **01** (remboursement ou échéance normale) : jamais un défaut.
- Codes **16** (titrisation de prêts redevenus performants) et **96** (défaut de souscription confirmé avant événement de crédit) : comptent seulement si le statut a déjà déclenché le défaut avant.
- Statut **XX** ou vide (non disponible) : traité comme manquant, jamais comme défaut.

Le défaut est **définitif** : un prêt qui redevient sain reste en défaut, daté du premier mois.

**Chiffres v1 :** millésime 2007 : 8 265 défauts sur 50 000 prêts (16,5 %, pic en 2009). Millésime 2015 : 2 058 défauts (4,1 %), dont 1 115 en 2020.

## Pourquoi v2 : le problème de 2020

Le millésime 2015 compte 1 115 défauts en 2020 sur 2 058. Presque tous portent un marqueur d'aide : un plan d'assistance (`Borrower Assistance Plan = F`, suspension de paiement) ou un retard dû à un sinistre (`Delinquency Due to Disaster = Y`). Le marqueur est présent sur **94,4 %** des défauts de 2020 du millésime 2015 et **87,1 %** de ceux du millésime 2007. Ces prêts sont comptés en défaut par la règle v1 parce que le paiement est suspendu, pas parce que l'emprunteur ne peut plus payer.

Taux de défaut cumulés selon le traitement :

| Millésime | Avec forbearance (v1) | Sans forbearance |
|---|---|---|
| 2007 | 16,53 % | 16,11 % |
| 2015 | 4,12 % | 1,34 % |

## Règle v2 (en vigueur)

Même règle que v1. Les défauts restent comptés mais sont **signalés** : la colonne `defaut_forbearance` est vraie si un marqueur F ou Y apparaît sur une ligne du prêt à la date du défaut ou avant. On peut alors calculer les taux avec et sans.

Dans la modélisation :
- **Run principal :** les modèles sont entraînés sur `defaut_hors_forbearance`.
- **Run de sensibilité :** entraînement sur `defaut` (forbearance comprise).

Le choix a un effet mesurable. Pour le millésime 2015, en 2021, la PD moyenne de la régression logistique est de 4,96 % si l'on entraîne avec la forbearance, contre 0,28 % sans (observé hors forbearance : 0,09 %). Le biais prudent de la logistique persiste jusqu'en 2026 dans le premier cas et disparaît dans le second.

## Limites

- « Hors forbearance » suppose que ces prêts ne sont pas de vrais défauts. Un prêt en forbearance peut ensuite faire défaut pour de bon : c'est une hypothèse de modélisation, pas une correction neutre.
- Aucune période de probation après un défaut n'est modélisée (un prêt redevenu sain reste en défaut).
- Le statut d'impayés est calculé à partir de la date du dernier échéancier payé, tel que rapporté par les gestionnaires ; il n'y a pas de montant d'impayé.
- La crise 2008–2010 n'est jamais observée hors échantillon (premières années de test : 2011 et 2019).
