


    
  
                     "recent_rotation_minutes_diff", "minutes_top5_share_diff",
                     "players_with_history_diff"]
MODELS = {
    "baseline_home_58": None,
    "elo": "elo",
    "logistic_form5": F5,
    "logistic_form10": F10,
    "logistic_form20": F20,
    "logistic_elo_form": FULL,
    "logistic_elo_player_form": FULL + PLAYER_FEATURES,
    "logistic_elo_rotation_form": FULL + PLAYER_FEATURES + ROTATION_FEATURES,
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
    rotation = build_player_rotation_features(players, games)
    features["game_id"] = features["game_id"].astype(str)
    rotation["game_id"] = rotation["game_id"].astype(str)
    features = features.merge(rotation, on="game_id", how="left", validate="one_to_one")
    for col in ROTATION_FEATURES:
        if col not in features:
            features[col] = 0.0
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)
    pred = predict(features)
    overall, by_season = summarize(pred)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "predictions.csv", index=False)
    overall.to_csv(out / "summary_overall.csv", index=False)
    by_season.to_csv(out / "summary_by_season.csv", index=False)
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = ["# NBA Oracle Backtest v3", "",
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
