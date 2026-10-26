#!/usr/bin/env python3
"""NBA Oracle Backtest v2: Elo + recent team form + player-box-score features.

The added features are rolling team aggregates of player box scores from prior
games only. They are not individual-player identity or injury/availability
features. Models are trained on seasons strictly earlier than the season tested.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
EXPECTED_GAMES = {"2021-22": 1230, "2022-23": 1230, "2023-24": 1230,
                  "2024-25": 1223, "2025-26": 1230}
BASE_URL = ("https://huggingface.co/datasets/cbratkovics/nba-game-logs/"
            "resolve/main/game_logs/game_logs_{season}.parquet")
KEYS = ["game_id", "game_date", "season", "team", "opponent", "home"]
PLAYER_STATS = ["pts", "reb", "ast", "minutes", "fgm", "fga", "fg3m", "fg3a",
                "ftm", "fta", "oreb", "dreb", "stl", "blk", "tov"]
ROLL_STATS = ["reb", "ast", "tov", "stl", "blk", "fgm", "fga", "fg3m", "fta", "ftm"]
ELO_K, ELO_HOME, ELO_SCALE = 20.0, 65.0, 400.0


def log(msg):
    print(f"[NBA ORACLE] {msg}", flush=True)


def download_season(season, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"game_logs_{season}.parquet"
    if path.exists() and path.stat().st_size > 10_000:
        return path
    url = BASE_URL.format(season=season)
    last_error = None
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                tmp = path.with_suffix(".parquet.part")
                with tmp.open("wb") as out:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            out.write(chunk)
                if tmp.stat().st_size < 10_000:
                    raise RuntimeError("Fichier téléchargé trop petit")
                tmp.replace(path)
                return path
        except Exception as exc:
            last_error = exc
            log(f"Téléchargement {season}, essai {attempt + 1}/3 : {exc}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Téléchargement impossible pour {season}: {last_error}")


def load_players(cache_dir):
    frames = []
    for season in SEASONS:
        path = download_season(season, cache_dir)
        df = pd.read_parquet(path)
        missing = set(KEYS + ["pts"]) - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: colonnes manquantes {sorted(missing)}")
        df["season"] = df["season"].astype(str)
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df[df["season"] == season].copy()
        for col in PLAYER_STATS:
            if col not in df:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df["home"] = df["home"].astype("boolean")
        frames.append(df)
        log(f"{season}: {len(df):,} lignes joueur-match")
    return pd.concat(frames, ignore_index=True).dropna(
        subset=["game_id", "game_date", "team", "opponent", "home", "pts"])


def reconstruct_games(players):
    team = players.groupby(KEYS, as_index=False, dropna=False)[PLAYER_STATS].sum()
    games, malformed = [], 0
    for game_id, group in team.groupby("game_id", sort=False):
        if len(group) != 2 or group["home"].nunique() != 2:
            malformed += 1
            continue
        h = group[group["home"] == True].iloc[0]
        a = group[group["home"] == False].iloc[0]
        if str(h["opponent"]) != str(a["team"]) or str(a["opponent"]) != str(h["team"]):
            malformed += 1
            continue
        rec = {
            "game_id": str(game_id), "season": str(h["season"]),
            "date": pd.Timestamp(h["game_date"]), "home_team": str(h["team"]),
            "away_team": str(a["team"]), "home_pts": int(h["pts"]),
            "away_pts": int(a["pts"]), "margin": int(h["pts"] - a["pts"]),
            "home_win": int(h["pts"] > a["pts"]),
        }
        for stat in PLAYER_STATS:
            rec[f"home_{stat}"] = float(h[stat])
            rec[f"away_{stat}"] = float(a[stat])
        games.append(rec)
    df = pd.DataFrame(games)
    if df.empty:
        raise RuntimeError("Aucun match valide reconstruit")
    df = df[df["season"].isin(SEASONS)].sort_values(
        ["date", "game_id"], kind="stable").drop_duplicates("game_id").reset_index(drop=True)
    audit = {
        "source": "cbratkovics/nba-game-logs on Hugging Face",
        "total_reconstructed_games": int(len(df)),
        "malformed_game_ids_excluded_count": int(malformed),
        "seasons": {s: {"expected_from_source_card": EXPECTED_GAMES[s],
                        "reconstructed_games": int((df.season == s).sum()),
                        "difference_vs_expected": int((df.season == s).sum() - EXPECTED_GAMES[s])}
                    for s in SEASONS},
        "limitations": [
            "Seven regular-season games are missing in the source for 2024-25.",
            "Scores are reconstructed by summing player points and are not independently verified against official box scores.",
            "Added features are rolling team aggregates of player box scores, not individual player identities or injury/availability data.",
            "No historical injuries, confirmed lineups, betting odds, shot charts, play-by-play or tracking data."
        ]
    }
    return df, audit


def mean(xs, default):
    return float(np.mean(xs)) if xs else float(default)


def build_features(games):
    elo = defaultdict(lambda: 1500.0)
    hist = defaultdict(lambda: deque(maxlen=20))
    last_date, rows = {}, []
    previous_season = None

    for g in games.itertuples(index=False):
        season = str(g.season)
        if season != previous_season:
            if previous_season is not None:
                for team in list(elo):
                    elo[team] = 1500 + 0.75 * (elo[team] - 1500)
            hist = defaultdict(lambda: deque(maxlen=20))
            last_date = {}
            previous_season = season

        h, a, date = g.home_team, g.away_team, pd.Timestamp(g.date)
        hh, aa = list(hist[h]), list(hist[a])
        hr = max(0, min((date - last_date[h]).days if h in last_date else 3, 10))
        ar = max(0, min((date - last_date[a]).days if a in last_date else 3, 10))
        r = {
            "game_id": g.game_id, "season": season, "date": date,
            "home_team": h, "away_team": a, "home_pts": g.home_pts,
            "away_pts": g.away_pts, "margin": g.margin, "home_win": g.home_win,
            "elo_diff": elo[h] + ELO_HOME - elo[a],
            "rest_diff": hr - ar, "home_b2b": int(hr <= 1), "away_b2b": int(ar <= 1),
        }
        for w in (5, 10, 20):
            H, A = hh[-w:], aa[-w:]
            r[f"home_margin_{w}"] = mean([x["margin"] for x in H], 0)
            r[f"away_margin_{w}"] = mean([x["margin"] for x in A], 0)
            r[f"home_winrate_{w}"] = mean([x["win"] for x in H], .5)
            r[f"away_winrate_{w}"] = mean([x["win"] for x in A], .5)
            r[f"home_pts_for_{w}"] = mean([x["pf"] for x in H], 110)
            r[f"away_pts_for_{w}"] = mean([x["pf"] for x in A], 110)
            r[f"home_pts_against_{w}"] = mean([x["pa"] for x in H], 110)
            r[f"away_pts_against_{w}"] = mean([x["pa"] for x in A], 110)

        # Features from previous completed games only. Current box score is appended below.
        for w in (5, 10):
            H, A = hh[-w:], aa[-w:]
            for stat in ROLL_STATS:
                r[f"{stat}_diff_{w}"] = (
                    mean([x.get(stat, 0) for x in H], 0) -
                    mean([x.get(stat, 0) for x in A], 0))
            he = mean([(x["fgm"] + .5*x["fg3m"]) / max(x["fga"], 1) for x in H], .5)
            ae = mean([(x["fgm"] + .5*x["fg3m"]) / max(x["fga"], 1) for x in A], .5)
            r[f"efg_diff_{w}"] = he - ae
            hf = mean([x["fta"] / max(x["fga"], 1) for x in H], .2)
            af = mean([x["fta"] / max(x["fga"], 1) for x in A], .2)
            r[f"ft_rate_diff_{w}"] = hf - af
        rows.append(r)

        expected = 1 / (1 + 10 ** (-(elo[h] + ELO_HOME - elo[a]) / ELO_SCALE))
        mult = math.log1p(abs(g.margin)) * (2.2 / (2.2 + .001 * abs(elo[h] - elo[a])))
        change = ELO_K * mult * (float(g.home_win) - expected)
        elo[h] += change
        elo[a] -= change

        def entry(team_prefix, is_home):
            get = lambda stat: float(getattr(g, f"{team_prefix}_{stat}"))
            return {
                "margin": int(g.margin if is_home else -g.margin),
                "win": int(g.home_win if is_home else 1-g.home_win),
                "pf": float(g.home_pts if is_home else g.away_pts),
                "pa": float(g.away_pts if is_home else g.home_pts),
                **{stat: get(stat) for stat in PLAYER_STATS}
            }
        hist[h].append(entry("home", True))
        hist[a].append(entry("away", False))
        last_date[h] = date
        last_date[a] = date
    return pd.DataFrame(rows)


F5 = ["home_margin_5","away_margin_5","home_winrate_5","away_winrate_5",
      "home_pts_for_5","away_pts_for_5","home_pts_against_5","away_pts_against_5",
      "rest_diff","home_b2b","away_b2b"]
F10 = [x.replace("_5", "_10") for x in F5[:8]] + F5[8:]
F20 = [x.replace("_5", "_20") for x in F5[:8]] + F5[8:]
FULL = ["elo_diff"] + F5[:8] + F10[:8] + F20[:8] + F5[8:]
PLAYER_FEATURES = [f"{s}_diff_{w}" for w in (5,10) for s in ROLL_STATS]
PLAYER_FEATURES += [f"efg_diff_{w}" for w in (5,10)] + [f"ft_rate_diff_{w}" for w in (5,10)]
MODELS = {
    "baseline_home_58": None,
    "elo": "elo",
    "logistic_form5": F5,
    "logistic_form10": F10,
    "logistic_form20": F20,
    "logistic_elo_form": FULL,
    "logistic_elo_player_form": FULL + PLAYER_FEATURES,
}


def predict(features):
    out = []
    for season in SEASONS:
        test = features[features.season == season]
        if test.empty:
            continue
        train = features[features.date < test.date.min()]
        log(f"{season}: {len(test)} matchs testés, {len(train)} matchs antérieurs pour l'entraînement")
        fitted = {}
        for name, cols in MODELS.items():
            if isinstance(cols, list) and len(train) >= 100 and train.home_win.nunique() == 2:
                m = make_pipeline(StandardScaler(), LogisticRegression(C=.5, max_iter=2000, random_state=42))
                m.fit(train[cols].astype(float), train.home_win.astype(int))
                fitted[name] = m
        for row in test.itertuples(index=False):
            for name, cols in MODELS.items():
                if name == "baseline_home_58":
                    p = .58
                elif name == "elo":
                    p = 1 / (1 + 10 ** (-float(row.elo_diff) / ELO_SCALE))
                elif name in fitted:
                    x = pd.DataFrame([{c: float(getattr(row, c)) for c in cols}])
                    p = float(fitted[name].predict_proba(x)[0, 1])
                else:
                    p = 1 / (1 + 10 ** (-float(row.elo_diff) / ELO_SCALE))
                p = min(max(p, 1e-6), 1-1e-6)
                out.append({"game_id": row.game_id, "season": season, "date": row.date,
                            "home_team": row.home_team, "away_team": row.away_team,
                            "actual_home_win": int(row.home_win), "model": name,
                            "p_home_win": p, "predicted_home_win": int(p >= .5)})
    return pd.DataFrame(out)


def summarize(pred):
    overall, by_season = [], []
    for name, g in pred.groupby("model"):
        y, p = g.actual_home_win.astype(int), g.p_home_win.astype(float)
        overall.append({"model": name, "games": len(g),
                        "accuracy": accuracy_score(y, g.predicted_home_win.astype(int)),
                        "brier": brier_score_loss(y, p),
                        "log_loss": log_loss(y, p, labels=[0,1])})
    for (season, name), g in pred.groupby(["season", "model"]):
        y, p = g.actual_home_win.astype(int), g.p_home_win.astype(float)
        by_season.append({"season": season, "model": name, "games": len(g),
                          "accuracy": accuracy_score(y, g.predicted_home_win.astype(int)),
                          "brier": brier_score_loss(y, p),
                          "log_loss": log_loss(y, p, labels=[0,1])})
    return (pd.DataFrame(overall).sort_values(["brier","log_loss"]),
            pd.DataFrame(by_season).sort_values(["season","model"]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/raw")
    ap.add_argument("--output", default="reports")
    args = ap.parse_args()
    players = load_players(Path(args.cache))
    games, audit = reconstruct_games(players)
    features = build_features(games)
    pred = predict(features)
    overall, by_season = summarize(pred)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "predictions.csv", index=False)
    overall.to_csv(out / "summary_overall.csv", index=False)
    by_season.to_csv(out / "summary_by_season.csv", index=False)
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = ["# NBA Oracle Backtest v2", "",
             f"Matchs reconstruits : {audit['total_reconstructed_games']}", "",
             "## Résultats globaux", "",
             "| Modèle | Matchs | Précision | Brier | Log loss |",
             "|---|---:|---:|---:|---:|"]
    for r in overall.itertuples(index=False):
        lines.append(f"| {r.model} | {r.games} | {r.accuracy:.3%} | {r.brier:.6f} | {r.log_loss:.6f} |")
    lines += ["", "## Limites", ""] + [f"- {x}" for x in audit["limitations"]]
    lines += ["", "Le modèle `logistic_elo_player_form` est la variante expérimentale. Compare-le au `logistic_elo_form` sur les mêmes matchs."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n=== RÉSULTATS GLOBAUX ===")
    print(overall.to_string(index=False))
    print(f"\nRapports écrits dans : {out.resolve()}")


if __name__ == "__main__":
    main()
