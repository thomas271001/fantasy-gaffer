# run_fpl_pipeline.py
import os
from pathlib import Path
import json, time, requests
import pandas as pd, numpy as np, joblib
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import pulp

DATA_DIR = Path("data")
PLAYER_SUMMARY_DIR = DATA_DIR / "player_summaries"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLAYER_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# Config
MAX_PLAYERS = None    # set to int for quick runs (Actions will use None)
PER_PLAYER_DELAY = 0.20

def fetch_bootstrap():
    URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    return r.json()

def build_history_with_fallback(boot, max_players=None, delay=0.2):
    players = boot.get("elements", [])
    teams = {t["id"]: t for t in boot.get("teams", [])}
    rows = []
    # try bootstrap history
    for p in players:
        for h in p.get("history", []):
            pid = p.get("id")
            name = p.get("web_name")
            team = teams.get(p.get("team"), {}).get("name", "Unknown")
            pos = p.get("element_type")
            cost_now = p.get("now_cost", 0) / 10.0
            gw = h.get("round")
            rows.append({
                "player_id": pid, "name": name, "team": team, "position": pos, "gw": gw,
                "minutes": h.get("minutes", 0), "goals_scored": h.get("goals_scored", 0),
                "assists": h.get("assists", 0), "clean_sheets": h.get("clean_sheets", 0),
                "total_points": h.get("total_points", 0), "value": cost_now
            })
    if len(rows) > 0:
        df = pd.DataFrame(rows)
        df.to_parquet(DATA_DIR / "player_gw_history.parquet", index=False)
        print(f"Built history from bootstrap for {df['player_id'].nunique():,} players and {len(df):,} rows.")
        return df

    # fallback: element-summary per player
    print("No per-player history in bootstrap. Fetching element-summary per player...")
    count = 0
    for p in players:
        pid = p.get("id")
        if max_players is not None and count >= max_players:
            break
        cache_file = PLAYER_SUMMARY_DIR / f"{pid}.json"
        if cache_file.exists():
            try:
                summary = json.loads(cache_file.read_text())
            except Exception:
                summary = None
        else:
            try:
                url = f"https://fantasy.premierleague.com/api/element-summary/{pid}/"
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                summary = r.json()
                cache_file.write_text(json.dumps(summary))
                time.sleep(delay)
            except Exception as e:
                print(f"Warning: failed to fetch element-summary for player {pid}: {e}")
                summary = None

        if not summary:
            continue

        for h in summary.get("history", []):
            name = p.get("web_name")
            team = teams.get(p.get("team"), {}).get("name", "Unknown")
            pos = p.get("element_type")
            cost_now = p.get("now_cost", 0) / 10.0
            gw = h.get("round")
            rows.append({
                "player_id": pid, "name": name, "team": team, "position": pos, "gw": gw,
                "minutes": h.get("minutes", 0), "goals_scored": h.get("goals_scored", 0),
                "assists": h.get("assists", 0), "clean_sheets": h.get("clean_sheets", 0),
                "total_points": h.get("total_points", 0), "value": cost_now
            })
        count += 1

    if len(rows) == 0:
        raise RuntimeError("No history rows were obtained. Aborting.")

    df = pd.DataFrame(rows)
    df.to_parquet(DATA_DIR / "player_gw_history.parquet", index=False)
    print(f"Fetched and built history for {df['player_id'].nunique():,} players; total rows: {len(df):,}.")
    return df

def feature_engineering_from_history(df, window=3):
    df = df.sort_values(["player_id", "gw"])
    def add_rolling(g):
        g = g.sort_values("gw")
        g["pts_roll3"] = g["total_points"].rolling(window, min_periods=1).mean().shift(1)
        g["min_roll3"] = g["minutes"].rolling(window, min_periods=1).mean().shift(1)
        g["goals_roll3"] = g["goals_scored"].rolling(window, min_periods=1).mean().shift(1)
        return g
    df_feats = df.groupby("player_id", group_keys=False).apply(add_rolling).reset_index(drop=True)
    df_feats = df_feats.dropna(subset=["pts_roll3"])
    df_feats.to_parquet(DATA_DIR / "player_features.parquet", index=False)
    return df_feats

def train_lgbm(df_feats):
    feature_cols = ["pts_roll3", "min_roll3", "goals_roll3", "value"]
    X = df_feats[feature_cols]
    y = df_feats["total_points"]
    tscv = TimeSeriesSplit(n_splits=5)
    maes, models = [], []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = lgb.LGBMRegressor(objective="regression", metric="mae", n_estimators=500)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="mae",
                  callbacks=[lgb.early_stopping(25, verbose=False)])
        preds = model.predict(X_val)
        maes.append(mean_absolute_error(y_val, preds))
        models.append(model)
    final_model = models[-1]
    joblib.dump(final_model, DATA_DIR / "lgb_model.pkl")
    return final_model, float(np.mean(maes))

def predict_next_gw(model, df_feats):
    feature_cols = ["pts_roll3", "min_roll3", "goals_roll3", "value"]
    latest = df_feats.sort_values("gw").groupby("player_id", group_keys=False).tail(1).reset_index(drop=True)
    X_latest = latest[feature_cols].fillna(0)
    preds = model.predict(X_latest)
    latest = latest.assign(predicted_points=preds)
    pred_df = latest[["player_id", "name", "team", "position", "value", "predicted_points", "gw"]]
    pred_df.to_csv(DATA_DIR / "predictions_next_gw.csv", index=False)
    return pred_df

def optimize_squad(pred_df, budget=100.0):
    players_df = pred_df.copy()
    players_df["position_name"] = players_df["position"].map({1:"GK",2:"DEF",3:"MID",4:"FWD"})
    players = players_df["player_id"].tolist()
    pred_map = players_df.set_index("player_id")["predicted_points"].to_dict()
    cost_map = players_df.set_index("player_id")["value"].to_dict()
    pos_map = players_df.set_index("player_id")["position_name"].to_dict()
    team_map = players_df.set_index("player_id")["team"].to_dict()

    model = pulp.LpProblem("FPL_15_player_squad", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", players, lowBound=0, upBound=1, cat="Integer")
    model += pulp.lpSum(pred_map[p] * x[p] for p in players)
    model += pulp.lpSum(cost_map[p] * x[p] for p in players) <= budget
    model += pulp.lpSum(x[p] for p in players) == 15
    for pos, cnt in [("GK",2),("DEF",5),("MID",5),("FWD",3)]:
        model += pulp.lpSum(x[p] for p in players if pos_map[p]==pos) == cnt
    for team in set(team_map.values()):
        model += pulp.lpSum(x[p] for p in players if team_map[p]==team) <= 3

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    selected = [p for p in players if pulp.value(x[p]) == 1]
    sel_df = players_df[players_df["player_id"].isin(selected)].sort_values("predicted_points", ascending=False)
    sel_df.to_csv(DATA_DIR / "selected_squad.csv", index=False)
    return sel_df

def main():
    print("Fetching bootstrap...")
    boot = fetch_bootstrap()
    with open(DATA_DIR / "bootstrap_static.json", "w") as f:
        json.dump(boot, f)
    print("Building history...")
    df_hist = build_history_with_fallback(boot, max_players=MAX_PLAYERS, delay=PER_PLAYER_DELAY)
    print("Rows:", len(df_hist))
    df_feats = feature_engineering_from_history(df_hist)
    print("Feature rows:", len(df_feats))
    model, cv_mae = train_lgbm(df_feats)
    print("CV MAE:", cv_mae)
    pred_df = predict_next_gw(model, df_feats)
    print("Predictions rows:", len(pred_df))
    sel_df = optimize_squad(pred_df)
    print("Selected rows:", len(sel_df))
    print("Wrote outputs to data/")

if __name__ == "__main__":
    main()
