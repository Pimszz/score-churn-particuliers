# Score de churn — clients particuliers (projet d'alternance)

Construction d'un modèle de prédiction de l'attrition (*churn*) des clients particuliers, de la donnée brute jusqu'au message métier. Projet mené durant mon alternance de M2 Statistique et Économie du Risque, au sein de l'équipe data d'une banque de détail.

> **Confidentialité.** Projet mené en contexte professionnel. Les noms de tables et de schémas internes, les codes produits, de classification et de réseau, ainsi que les identifiants clients, ont été remplacés ou masqués ; les taux de churn et les volumes sensibles ne sont pas communiqués. Le code illustre la démarche et n'est pas exécutable en l'état, faute d'accès aux données.

## Problématique

Peut-on, à partir des données clients, à la fois **expliquer** les facteurs de départ des clients particuliers établis et **prédire** ce risque de façon assez fiable pour cibler des actions de rétention ? Un double objectif : un modèle interprétable pour comprendre, un modèle performant pour agir.

## Démarche

**1. Construction de la donnée (SQL, Snowflake).** Une table analytique par client, avec une date de référence paramétrable, agrégeant quatre familles de variables : profil et contactabilité, détention de produits, encours d'épargne, comportement transactionnel et digital. Environ 58 variables explicatives au final.

**2. Traitement des valeurs extrêmes, cas par cas.** Plutôt qu'une correction mécanique, une investigation : un client cumulant plusieurs millions de connexions web sur l'année (un robot, plafonné), des z-scores aberrants dus à un écart-type quasi nul. La règle retenue : comprendre l'origine d'une valeur extrême avant de décider de son traitement.

**3. Sélection des variables.** Tests univariés (Mann-Whitney, chi-deux) à titre exploratoire, contrôle de la colinéarité par familles temporelles, sélection effective portée par la régularisation et les modèles.

**4. Deux modèles complémentaires.**
- **Régression logistique Elastic Net** — interprétable, lecture par *Odds Ratios*. AUC ≈ 0,726.
- **LightGBM** (optimisé par Optuna) — capte les non-linéarités et les interactions. AUC ≈ 0,783. **Modèle retenu.**

**5. Validation dans le temps (*out-of-time*).** Entraînement sur une année, test sur l'année suivante, jamais vue. AUC ≈ 0,771 : la faible perte de performance atteste d'une robustesse temporelle.

**6. Du score à l'action.** Choix d'un seuil de décision (0,25) sur la courbe précision-rappel. À ce point, le modèle détecte environ **un départ sur deux**, avec un **lift ×3,2** par rapport à un ciblage aléatoire. Principaux enseignements métier : l'équipement en produits est le premier levier de rétention (la détention d'un crédit immobilier protège fortement, *OR* ≈ 0,27), tandis que le statut de mono-détenteur accroît le risque (*OR* ≈ 1,22).

## Ce que ce projet illustre

Une chaîne complète, du SQL brut au message actionnable pour le métier : ingénierie de la donnée, modélisation statistique et *machine learning*, validation rigoureuse, et traduction des résultats en leviers concrets.

## Stack technique

SQL (Snowflake) · Python (pandas, scikit-learn, LightGBM, Optuna) · matplotlib

## Contenu du dépôt

- 'construction_table.sql' — requête de construction de la table analytique (anonymisée).
- 'pipeline_modelisation.py' — extrait anonymisé du pipeline de modélisation.
