"""Build compact game + player tables from hoopR parquet files (local clone or downloaded)."""
import sys, pandas as pd, numpy as np
SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/sportsdataverse/hoopr-mbb-data/mbb"
SEASONS = range(2018, 2028)

def team_rows():
    out = []
    for y in SEASONS:
        t = pd.read_parquet(f"{SRC}/team_box/parquet/team_box_{y}.parquet")
        out.append(t)
    return pd.concat(out, ignore_index=True)

def main():
    t = team_rows()
    t = t.rename(columns={"total_turnovers": "tov"})
    keep = ["game_id","season","game_date","team_id","team_display_name","team_home_away","team_score",
            "field_goals_made","field_goals_attempted","three_point_field_goals_made","three_point_field_goals_attempted",
            "free_throws_made","free_throws_attempted","offensive_rebounds","defensive_rebounds","tov","opponent_team_id"]
    t = t[keep].copy()
    t["game_date"] = pd.to_datetime(t["game_date"])
    for c in keep[6:-1]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.dropna(subset=["team_score","field_goals_attempted","tov","offensive_rebounds","free_throws_attempted"])
    t["team_id"] = t.team_id.astype(int); t["opponent_team_id"] = t.opponent_team_id.astype(int)
    # one row per game with both sides
    h = t[t.team_home_away == "home"].set_index("game_id"); a = t[t.team_home_away == "away"].set_index("game_id")
    g = h.join(a, lsuffix="_h", rsuffix="_a", how="inner").reset_index()
    sched = []
    for y in SEASONS:
        s = pd.read_parquet(f"{SRC}/schedules/parquet/mbb_schedule_{y}.parquet", columns=["game_id","neutral_site","season_type","status_type_completed","home_id","away_id","home_display_name","away_display_name","game_date","conference_competition"])
        sched.append(s)
    s = pd.concat(sched); s["game_id"] = s.game_id.astype(str); g["game_id"] = g.game_id.astype(str)
    g = g.merge(s[["game_id","neutral_site","season_type","conference_competition"]].drop_duplicates("game_id"), on="game_id", how="left")
    g["neutral_site"] = g.neutral_site.fillna(False).astype(bool)
    g = g.rename(columns={"season_h":"season","game_date_h":"date","team_id_h":"hid","team_id_a":"aid","team_display_name_h":"hname","team_display_name_a":"aname"})
    # possessions per team estimate (standard), game poss = mean of the two
    for x in ("h","a"):
        g[f"poss_{x}"] = g[f"field_goals_attempted_{x}"] - g[f"offensive_rebounds_{x}"] + g[f"tov_{x}"] + 0.475*g[f"free_throws_attempted_{x}"]
    g["poss"] = (g.poss_h + g.poss_a)/2
    g = g[(g.poss > 40) & (g.poss < 110)]
    # drop teams with too few games in a season (non-D1)
    cnt = pd.concat([g[["season","hid"]].rename(columns={"hid":"t"}), g[["season","aid"]].rename(columns={"aid":"t"})]).value_counts().rename("n").reset_index()
    ok = set(map(tuple, cnt[cnt.n >= 5][["season","t"]].values))
    cur = cnt.season.max()   # in-progress season: few games so far, so also accept teams that were D1 last season
    prev = {t_ for s_, t_ in ok if s_ == cur - 1}
    ok |= {(cur, t_) for t_ in cnt[cnt.season == cur].t if t_ in prev}
    g = g[[ (s_,h_) in ok and (s_,a_) in ok for s_,h_,a_ in zip(g.season,g.hid,g.aid)]]
    cols = ["game_id","season","date","hid","aid","hname","aname","neutral_site","season_type","poss","poss_h","poss_a"] + \
        [f"{c}_{x}" for x in ("h","a") for c in ["team_score","field_goals_made","field_goals_attempted","three_point_field_goals_made","three_point_field_goals_attempted","free_throws_made","free_throws_attempted","offensive_rebounds","defensive_rebounds","tov"]]
    g = g[cols].sort_values("date").reset_index(drop=True)
    g.to_parquet("data/games.parquet"); print("games", g.shape, g.groupby("season").size().to_dict())
    # players
    P = []
    for y in SEASONS:
        p = pd.read_parquet(f"{SRC}/player_box/parquet/player_box_{y}.parquet", columns=["game_id","season","game_date","athlete_id","athlete_display_name","team_id","minutes","points","field_goals_made","field_goals_attempted","three_point_field_goals_made","free_throws_made","free_throws_attempted","offensive_rebounds","defensive_rebounds","assists","steals","blocks","turnovers","fouls","starter","did_not_play"])
        P.append(p)
    p = pd.concat(P, ignore_index=True)
    p["game_id"] = p.game_id.astype(str); p["game_date"] = pd.to_datetime(p.game_date)
    p = p[p.game_id.isin(set(g.game_id))].copy()
    p["minutes"] = pd.to_numeric(p.minutes, errors="coerce").fillna(0)
    p["did_not_play"] = p.did_not_play.astype(bool)
    p = p[~p.did_not_play | (p.minutes > 0)]
    p["athlete_id"] = pd.to_numeric(p.athlete_id, errors="coerce"); p = p.dropna(subset=["athlete_id"]); p["athlete_id"] = p.athlete_id.astype("int64"); p["team_id"] = p.team_id.astype(int)
    p.to_parquet("data/players.parquet"); print("players", p.shape)
main()
