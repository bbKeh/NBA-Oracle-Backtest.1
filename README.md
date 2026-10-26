# NBA Oracle — Backtest historique sur 5 saisons

Ce dépôt est un **module autonome de backtest**, conçu pour être ajouté à NBA Oracle sans modifier son moteur actuel.

Il télécharge automatiquement les logs de matchs publics des saisons régulières 2021-22 à 2025-26, reconstruit les scores d'équipe à partir des lignes joueur-match, puis évalue plusieurs approches de prédiction en respectant l'ordre chronologique.

## Ce que le test fait

- Récupère les cinq fichiers Parquet depuis Hugging Face.
- Vérifie les colonnes, les saisons et le nombre de matchs reconstruits.
- Reconstruit un match par `game_id` à partir des points de chaque équipe.
- Calcule les caractéristiques **avant** chaque match : Elo, forme récente, points marqués/encaissés, bilan récent et repos.
- Évalue plusieurs variantes :
  - référence « avantage domicile » (probabilité fixe de 58 %, à comparer et non à considérer comme vérité universelle) ;
  - Elo avec avantage domicile ;
  - régression logistique basée sur la forme des 5 derniers matchs ;
  - régression logistique basée sur la forme des 10 derniers matchs ;
  - régression logistique basée sur la forme des 20 derniers matchs ;
  - régression logistique combinant Elo et plusieurs indicateurs de forme.
- Pour chaque saison, entraîne les modèles supervisés uniquement sur les saisons antérieures. La saison 2021-22 est une saison de démarrage à froid : les modèles appris utilisent Elo jusqu'à ce qu'une saison historique soit disponible pour l'entraînement.
- Produit les scores de précision, Brier, log loss et les prédictions match par match.
- Enregistre les données téléchargées dans le cache de la machine d'exécution, mais ne les ajoute pas au dépôt.

## Limites importantes

- La source est un jeu de données de logs joueurs, pas une base officielle complète des résultats d'équipe. Les scores sont reconstruits en additionnant les points des joueurs présents dans le fichier.
- La source signale **7 matchs manquants en 2024-25**. Le programme ne les invente pas et indique les écarts dans le rapport.
- Le jeu de données ne fournit pas les shot charts, le play-by-play complet, les blessures pré-match, les compositions attendues, les cotes historiques ou les données de tracking. Ce backtest ne valide donc pas à lui seul toutes les dimensions du protocole NBA Oracle.
- Une meilleure performance historique ne prouve pas une meilleure performance future. Les résultats doivent être examinés par saison et comparés à des références simples.
- La saison 2021-22 ne bénéficie pas d'une saison antérieure dans cette source. Ses résultats sont rapportés séparément comme démarrage à froid.
- Les résultats sont ceux d'un modèle historique expérimental, pas des garanties ni des conseils de pari.

## Lancer dans GitHub

1. Crée un dépôt GitHub, par exemple `nba-oracle-backtest`.
2. Téléverse le contenu de ce dossier à la racine du dépôt (y compris le dossier `.github/workflows`).
3. Ouvre l'onglet **Actions**.
4. Sélectionne **NBA Oracle — Backtest 5 saisons**, puis **Run workflow**.
5. À la fin, ouvre l'exécution et télécharge l'artefact `nba-oracle-backtest-report`.

Le workflow peut aussi être déclenché à chaque modification de `main.py`, du script de backtest ou du workflow lui-même.

## Fichiers de sortie

- `summary_by_season.csv` : métriques par saison et modèle.
- `summary_overall.csv` : métriques agrégées.
- `predictions.csv` : une ligne par match et modèle, avec probabilités et résultat réel.
- `data_audit.json` : nombre de matchs par saison, doublons et contrôles de qualité.
- `REPORT.md` : résumé lisible.

## Données source

Hugging Face : https://huggingface.co/datasets/cbratkovics/nba-game-logs

Le script utilise les fichiers `game_logs/game_logs_YYYY-YY.parquet` publiés dans le dépôt de données.
