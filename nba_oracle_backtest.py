#!/usr/bin/env python3
"""NBA Oracle: chronological five-season historical backtest.

Important: model fitting for a target season uses only games from earlier seasons.
Within a season, team-form and Elo features update after every completed game,
but supervised model coefficients remain fixed until the next season.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
EXPECTED_GAMES = {
    "2021-22": 1230,
    "2022-23": 1230,
    "2023-24": 1230,
    "2024-25": 1223,  # source documents seven missing games
    "2025-26": 1230,
}
BASE_URL = (
    "https://huggingface.co/datasets/cbratkovics/nba-game-logs/"
    "resolve/main/game_logs/game_logs_{season}.parquet"
)
REQUIRED_COLUMNS = {
    "game_id", "game_date", "season", "team", "opponent", "home", "pts"
}
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 65.0
ELO_SCALE = 400.0
COLD_START_HOME_PROB = 0.58


def log(message: str) -> None:
    print(f"[NBA ORACLE] {message}", flush=True)


def download_season(season: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"game_logs_{season}.parquet"
    if path.exists() and path.stat().st_size > 10_000:
        log(f"Cache présent : {path.name}")
        return path

    url = BASE_URL.format(season=season)
    log(f"Téléchargement {season} depuis Hugging Face…")
    last_error = None
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(20, 120), allow_redirects=True) as response:
                response.raise_for_status()
                temp = path.with_suffix(".parquet.part")
                total = 0
                with temp.open("wb") as out:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            out.write(chunk)
                            total += len(chunk)
                if total < 10_000:
                    raise RuntimeError(f"Fichier téléchargé trop petit ({total} octets)")
                temp.replace(path)
                return path
        except Exception as exc:
            last_error = exc
            log(f"Essai {attempt + 1}/3 échoué : {exc}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Impossible de télécharger {season}: {last_error}")


def load_player_logs(cache_dir: Path) -> pd.DataFrame:
    parts = []
    for season in SEASONS:
        path = download_season(season, cache_dir)
        frame = pd.read_parquet(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name}: colonnes manquantes: {sorted(missing)}")
        frame = frame.copy()
        frame["season"] = frame["season"].astype(str)
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
        frame["pts"] = pd.to_numeric(frame["pts"], errors="coerce")
        frame["home"] = frame["home"].astype("boolean")
        frame = frame[frame["season"] == season]
        parts.append(frame)
        log(f"{season}: {len(frame):,} lignes joueur-match")
    data = pd.concat(parts, ignore_index=True)
    data = data.dropna(subset=["game_id", "game_date", "team", "opponent", "home", "pts"])
    data["pts"] = data["pts"].astype(int)
    return data


def reconstruct_games(players: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Player rows are aggregated into one row per team per game.
    team_games = (
        players.groupby(
            ["game_id", "season", "game_date", "team", "opponent", "home"],
            as_index=False,
            dropna=False,
        )["pts"].sum()
    )
    audit = {
        "source": "cbratkovics/nba-game-logs on Hugging Face",
        "seasons": {},
        "known_source_limitations": [
            "Seven regular-season games missing in 2024-25 according to source card.",
            "Player DNP rows are excluded; team scores are reconstructed by summing player points.",
            "No historical injuries, odds, shot charts, play-by-play or tracking data in this dataset."
        ],
        "reconstruction_method": "sum player points by game_id and team; pair home and away teams",
    }

    games = []
    malformed = []
    for game_id, group in team_games.groupby("game_id", sort=False):
        if len(group) != 2 or group["home"].nunique() != 2:
            malformed.append(str(game_id))
            continue
        home_row = group[group["home"] == True].iloc[0]
        away_row = group[group["home"] == False].iloc[0]
        if str(home_row["opponent"]) != str(away_row["team"]) or str(away_row["opponent"]) != str(home_row["team"]):
            malformed.append(str(game_id))
            continue
        games.append({
            "game_id": str(game_id),
            "season": str(home_row["season"]),
            "date": pd.Timestamp(home_row["game_date"]),
            "home_team": str(home_row["team"]),
            "away_team": str(away_row["team"]),
            "home_pts": int(home_row["pts"]),
            "away_pts": int(away_row["pts"]),
            "margin": int(home_row["pts"]) - int(away_row["pts"]),
            "home_win": int(home_row["pts"] > away_row["pts"]),
        })

    result = pd.DataFrame(games)
    if result.empty:
        raise RuntimeError("Aucun match valide n'a pu être reconstruit.")
    result = result[result["season"].isin(SEASONS)].copy()
    result = result.sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)

    duplicate_ids = result["game_id"].duplicated().sum()
    if duplicate_ids:
        result = result.drop_duplicates("game_id", keep="first").reset_index(drop=True)

    for season in SEASONS:
        season_games = result[result["season"] == season]
        audit["seasons"][season] = {
            "expected_from_source_card": EXPECTED_GAMES[season],
            "reconstructed_games": int(len(season_games)),
            "difference_vs_expected": int(len(season_games) - EXPECTED_GAMES[season]),
            "team_pairs_validated": True,
        }
    audit["malformed_game_ids_excluded_count"] = len(malformed)
    audit["malformed_game_ids_excluded_sample"] = malformed[:30]
    audit["duplicate_game_ids_removed"] = int(duplicate_ids)
    audit["total_reconstructed_games"] = int(len(result))
    audit["score_reconstruction_warning"] = (
        "Scores are sums of player points from this source; this script cannot independently "
        "verify every score against an official box score."
    )
    return result, audit


def safe_mean(values, default):
    return float(np.mean(values)) if values else float(default)


def build_pregame_features(games: pd.DataFrame) -> pd.DataFrame:
    """Build features strictly before each game, then update state with its result."""
    elo = defaultdict(lambda: 1500.0)
    history = defaultdict(lambda: deque(maxlen=20))
    last_date = {}
    records = []
    current_season = None

    for row in games.itertuples(index=False):
        season = str(row.season)
        if season != current_season:
            # Regress Elo ratings modestly toward league average at season boundary.
            if current_season is not None:
                for team in list(elo):
                    elo[team] = 1500.0 + 0.75 * (elo[team] - 1500.0)
            # Do not carry recent form across the offseason/roster-change boundary.
            history = defaultdict(lambda: deque(maxlen=20))
            last_date = {}
            current_season = season

        h, a = row.home_team, row.away_team
        h_hist, a_hist = list(history[h]), list(history[a])
        date = pd.Timestamp(row.date)

        h_rest = (date - last_date[h]).days if h in last_date else 3
        a_rest = (date - last_date[a]).days if a in last_date else 3
        h_rest = max(0, min(int(h_rest), 10))
        a_rest = max(0, min(int(a_rest), 10))

        features = {
            "game_id": row.game_id,
            "season": season,
            "date": date,
            "home_team": h,
            "away_team": a,
            "home_pts": int(row.home_pts),
            "away_pts": int(row.away_pts),
            "margin": int(row.margin),
            "home_win": int(row.home_win),
            "elo_diff": (elo[h] + ELO_HOME_ADVANTAGE) - elo[a],
            "home_rest": h_rest,
            "away_rest": a_rest,
            "rest_diff": h_rest - a_rest,
            "home_b2b": int(h_rest <= 1),
            "away_b2b": int(a_rest <= 1),
            "home_prior_games": len(h_hist),
            "away_prior_games": len(a_hist),
        }

        for window in (5, 10, 20):
            hh, aa = h_hist[-window:], a_hist[-window:]
            features[f"home_margin_{window}"] = safe_mean([x["margin"] for x in hh], 0.0)
            features[f"away_margin_{window}"] = safe_mean([x["margin"] for x in aa], 0.0)
            features[f"home_winrate_{window}"] = safe_mean([x["win"] for x in hh], 0.5)
            features[f"away_winrate_{window}"] = safe_mean([x["win"] for x in aa], 0.5)
            features[f"home_pts_for_{window}"] = safe_mean([x["pf"] for x in hh], 110.0)
            features[f"away_pts_for_{window}"] = safe_mean([x["pf"] for x in aa], 110.0)
            features[f"home_pts_against_{window}"] = safe_mean([x["pa"] for x in hh], 110.0)
            features[f"away_pts_against_{window}"] = safe_mean([x["pa"] for x in aa], 110.0)

        records.append(features)

        expected_home = 1.0 / (1.0 + 10.0 ** (-(elo[h] + ELO_HOME_ADVANTAGE - elo[a]) / ELO_SCALE))
        actual_home = float(row.home_win)
        margin_abs = abs(int(row.margin))
        # Modest margin multiplier to make Elo respond more to decisive results.
        multiplier = math.log1p(margin_abs) * (2.2 / (2.2 + 0.001 * abs(elo[h] - elo[a])))
        change = ELO_K * multiplier * (actual_home - expected_home)
        elo[h] += change
        elo[a] -= change

        history[h].append({"margin": int(row.margin), "win": int(row.home_win), "pf": int(row.home_pts), "pa": int(row.away_pts)})
        history[a].append({"margin": -int(row.margin), "win": 1 - int(row.home_win), "pf": int(row.away_pts), "pa": int(row.home_pts)})
        last_date[h] = date
        last_date[a] = date

    return pd.DataFrame(records)


FEATURES_5 = [
    "home_margin_5", "away_margin_5", "home_winrate_5", "away_winrate_5",
    "home_pts_for_5", "away_pts_for_5", "home_pts_against_5", "away_pts_against_5",
    "rest_diff", "home_b2b", "away_b2b",
]
FEATURES_10 = [
    "home_margin_10", "away_margin_10", "home_winrate_10", "away_winrate_10",
    "home_pts_for_10", "away_pts_for_10", "home_pts_against_10", "away_pts_against_10",
    "rest_diff", "home_b2b", "away_b2b",
]
FEATURES_20 = [
    "home_margin_20", "away_margin_20", "home_winrate_20", "away_winrate_20",
    "home_pts_for_20", "away_pts_for_20", "home_pts_against_20", "away_pts_against_20",
    "rest_diff", "home_b2b", "away_b2b",
]
FEATURES_FULL = [
    "elo_diff", "home_margin_5", "away_margin_5", "home_winrate_5", "away_winrate_5",
    "home_pts_for_5", "away_pts_for_5", "home_pts_against_5", "away_pts_against_5",
    "home_margin_10", "away_margin_10", "home_winrate_10", "away_winrate_10",
    "home_pts_for_10", "away_pts_for_10", "home_pts_against_10", "away_pts_against_10",
    "home_margin_20", "away_margin_20", "home_winrate_20", "away_winrate_20",
    "home_pts_for_20", "away_pts_for_20", "home_pts_against_20", "away_pts_against_20",
    "rest_diff", "home_b2b", "away_b2b",
]

MODELS = {
    "baseline_home_58": None,
    "elo": "elo",
    "logistic_form5": FEATURES_5,
    "logistic_form10": FEATURES_10,
    "logistic_form20": FEATURES_20,
    "logistic_elo_form": FEATURES_FULL,
}


def elo_probability(elo_diff):
    return float(1.0 / (1.0 + 10.0 ** (-float(elo_diff) / ELO_SCALE)))


def fit_predict_by_season(features: pd.DataFrame) -> pd.DataFrame:
    predictions = []
    for season in SEASONS:
        season_rows = features[features["season"] == season].copy()
        prior_rows = features[features["date"] < season_rows["date"].min()].copy()
        log(f"Évaluation {season}: {len(season_rows)} matchs; entraînement sur {len(prior_rows)} matchs antérieurs.")

        fitted = {}
        for model_name, cols in MODELS.items():
            if cols is None or cols == "elo":
                continue
            # Require both outcome classes and a reasonable historical sample.
            if len(prior_rows) >= 100 and prior_rows["home_win"].nunique() == 2:
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=0.5, max_iter=2000, random_state=42),
                )
                model.fit(prior_rows[cols].astype(float), prior_rows["home_win"].astype(int))
                fitted[model_name] = model

        for row in season_rows.itertuples(index=False):
            record_base = {
                "game_id": row.game_id, "season": row.season, "date": row.date,
                "home_team": row.home_team, "away_team": row.away_team,
                "home_pts": row.home_pts, "away_pts": row.away_pts,
                "actual_home_win": int(row.home_win),
                "actual_margin": int(row.margin),
            }
            for model_name, cols in MODELS.items():
                if model_name == "baseline_home_58":
                    p = COLD_START_HOME_PROB
                    method = "fixed_home_probability"
                elif model_name == "elo":
                    p = elo_probability(row.elo_diff)
                    method = "elo"
                elif model_name in fitted:
                    model = fitted[model_name]
                    x = pd.DataFrame([{col: float(getattr(row, col)) for col in cols}], columns=cols)
                    p = float(model.predict_proba(x)[0, 1])
                    method = "logistic_trained_on_prior_seasons"
                else:
                    # First season cold start: no prior-season labels are available.
                    p = elo_probability(row.elo_diff)
                    method = "elo_cold_start_no_prior_season"
                p = min(max(p, 1e-6), 1 - 1e-6)
                predictions.append({
                    **record_base,
                    "model": model_name,
                    "p_home_win": p,
                    "predicted_home_win": int(p >= 0.5),
                    "method": method,
                    "training_games_before_season": int(len(prior_rows)),
                })
    return pd.DataFrame(predictions)


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (season, model), group in predictions.groupby(["season", "model"], sort=True):
        y = group["actual_home_win"].astype(int).to_numpy()
        p = group["p_home_win"].astype(float).to_numpy()
        yhat = group["predicted_home_win"].astype(int).to_numpy()
        rows.append({
            "season": season,
            "model": model,
            "games": int(len(group)),
            "accuracy": float(accuracy_score(y, yhat)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "mean_predicted_home_probability": float(np.mean(p)),
            "actual_home_win_rate": float(np.mean(y)),
            "margin_mae_na": np.nan,  # no margin regressor in this first release
        })
    by_season = pd.DataFrame(rows)

    overall_rows = []
    for model, group in predictions.groupby("model", sort=True):
        y = group["actual_home_win"].astype(int).to_numpy()
        p = group["p_home_win"].astype(float).to_numpy()
        yhat = group["predicted_home_win"].astype(int).to_numpy()
        overall_rows.append({
            "model": model,
            "games": int(len(group)),
            "accuracy": float(accuracy_score(y, yhat)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "mean_predicted_home_probability": float(np.mean(p)),
            "actual_home_win_rate": float(np.mean(y)),
        })
    overall = pd.DataFrame(overall_rows).sort_values(["brier", "log_loss"]).reset_index(drop=True)
    return by_season, overall


def write_report(output_dir: Path, audit: dict, by_season: pd.DataFrame, overall: pd.DataFrame, predictions: pd.DataFrame):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    by_season.to_csv(output_dir / "summary_by_season.csv", index=False)
    overall.to_csv(output_dir / "summary_overall.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    lines = [
        "# NBA Oracle — rapport du backtest",
        "",
        "## Statut",
        "",
        "Rapport généré par le workflow. Les résultats ne sont pas pré-calculés ni inventés : ils proviennent de l'exécution du script sur les fichiers téléchargés.",
        "",
        "## Qualité des données",
        "",
        f"- Matchs reconstruits : **{audit.get('total_reconstructed_games', 0)}**.",
        f"- Identifiants de match mal formés exclus : **{audit.get('malformed_game_ids_excluded_count', 0)}**.",
        f"- Doublons retirés : **{audit.get('duplicate_game_ids_removed', 0)}**.",
        "",
        "| Saison | Attendus selon la source | Reconstruits | Écart |",
        "|---|---:|---:|---:|",
    ]
    for season, values in audit["seasons"].items():
        lines.append(
            f"| {season} | {values['expected_from_source_card']} | "
            f"{values['reconstructed_games']} | {values['difference_vs_expected']:+d} |"
        )
    lines += [
        "",
        "## Résultats globaux",
        "",
        "| Modèle | Matchs | Précision | Brier (plus bas = mieux) | Log loss (plus bas = mieux) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in overall.itertuples(index=False):
        lines.append(
            f"| {r.model} | {r.games} | {r.accuracy:.3%} | {r.brier:.4f} | {r.log_loss:.4f} |"
        )
    lines += [
        "",
        "## Précautions d'interprétation",
        "",
        "- La saison 2021-22 est un démarrage à froid, car aucune saison antérieure n'est incluse dans la source.",
        "- Pour chaque saison, les modèles supervisés ne sont entraînés que sur les saisons antérieures. Les statistiques de forme et Elo d'un match sont calculés avant ce match.",
        "- Le script ne dispose pas des blessures pré-match, compositions, cotes historiques, shot charts, play-by-play ni données de tracking.",
        "- Les scores sont reconstruits à partir des points individuels. Vérifier les écarts du rapport avant d'interpréter les métriques.",
        "- Une meilleure performance historique ne prouve pas une meilleure prédiction future. Examiner la stabilité par saison et l'écart face aux références.",
        "",
        "## Fichiers",
        "",
        "- `summary_by_season.csv` — métriques par saison et modèle.",
        "- `summary_overall.csv` — métriques agrégées.",
        "- `predictions.csv` — toutes les prédictions match par match.",
        "- `data_audit.json` — audit de la source et de la reconstruction.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/raw", help="Dossier de cache des fichiers Parquet")
    parser.add_argument("--output", default="reports", help="Dossier des rapports")
    args = parser.parse_args()
    cache_dir = Path(args.cache)
    output_dir = Path(args.output)

    log("Début du backtest historique NBA Oracle.")
    log("Aucun résultat futur n'est utilisé dans les variables pré-match.")
    players = load_player_logs(cache_dir)
    games, audit = reconstruct_games(players)
    features = build_pregame_features(games)
    predictions = fit_predict_by_season(features)
    by_season, overall = summarize(predictions)
    write_report(output_dir, audit, by_season, overall, predictions)
    log(f"Terminé : {len(games)} matchs reconstruits, {len(predictions)} prédictions modèle.")
    log(f"Rapport principal : {output_dir / 'REPORT.md'}")
    print("\n=== RÉSULTATS GLOBAUX ===")
    print(overall.to_string(index=False))
    print("\n=== QUALITÉ DES DONNÉES ===")
    for season, details in audit["seasons"].items():
        print(f"{season}: {details['reconstructed_games']} matchs (attendus {details['expected_from_source_card']})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"ÉCHEC: {exc}")
        raise
