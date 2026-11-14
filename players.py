"""Player-level layer: per-player impact (game score per 40, shrunk) and 'missing value' for absent rotation players."""
import numpy as np, pandas as pd

def load_players():
    p = pd.read_parquet("data/players.parquet")
    p["game_date"] = pd.to_datetime(p.game_date)
    for c in ["points","field_goals_made","field_goals_attempted","free_throws_made","free_throws_attempted","offensive_rebounds","defensive_rebounds","assists","steals","blocks","turnovers","fouls"]:
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0)
    p["gs"] = (p.points + 0.4*p.field_goals_made - 0.7*p.field_goals_attempted - 0.4*(p.free_throws_attempted - p.free_throws_made)
               + 0.7*p.offensive_rebounds + 0.3*p.defensive_rebounds + p.steals + 0.7*p.assists + 0.7*p.blocks - 0.4*p.fouls - p.turnovers)
    return p

REPL = 7.0        # replacement-level game score per 40 min (calibrated below from <15 mpg players)
PRIOR_MIN = 120   # minutes of shrinkage toward replacement/team-typical

def team_season_table(p_ts, games_dates, repl=REPL):
    """p_ts: player-games for one team-season. Returns per team-game (date order): list of (athlete, rot_min, v40) known BEFORE that game."""
    pm = p_ts.pivot_table(index="game_date", columns="athlete_id", values="minutes", aggfunc="sum").reindex(games_dates).fillna(0)
    pg = p_ts.pivot_table(index="game_date", columns="athlete_id", values="gs", aggfunc="sum").reindex(games_dates).fillna(0)
    rm = pm.rolling(10, min_periods=1).mean().shift(1)              # recent avg minutes (incl. zeros when absent)
    cm = pm.cumsum().shift(1); cg = pg.cumsum().shift(1)
    v40 = (cg + PRIOR_MIN / 40 * repl) / (cm + PRIOR_MIN) * 40 if False else (cg / 40 * 40 + PRIOR_MIN / 40 * repl) / ((cm + PRIOR_MIN) / 40)
    return pm, rm, v40

def missing_values(p, games):
    """For every team-game: sum over rotation players (recent avg >= 14 mpg) who logged 0 minutes of (rm/40)*(v40-repl)."""
    out = []
    pp = p[["season","team_id","athlete_id","game_date","minutes","gs"]]
    for (s, t), x in pp.groupby(["season","team_id"]):
        dates = np.sort(x.game_date.unique())
        pm, rm, v40 = team_season_table(x, dates)
        rot = (rm >= 14)
        absent = rot & (pm == 0)
        val = (rm / 40 * (v40 - REPL)).where(absent, 0).sum(axis=1)
        nabs = absent.sum(axis=1)
        out.append(pd.DataFrame({"season": s, "team_id": t, "date": dates, "miss_val": val.values, "n_abs": nabs.values}))
    return pd.concat(out, ignore_index=True)

if __name__ == "__main__":
    p = load_players()
    low = p[(p.minutes >= 5)].groupby(["season","athlete_id"]).agg(m=("minutes","sum"), g=("gs","sum"), n=("game_id","count"))
    low["mpg"] = low.m / low.n; low["gs40"] = low.g / low.m * 40
    print("gs40 by mpg bucket:", low.groupby(pd.cut(low.mpg, [0,10,15,20,25,30,45])).gs40.mean().round(2).to_dict())
    mv = missing_values(p, None)
    mv.to_parquet("data/missing.parquet"); print(mv.describe().round(2))
