"""Daily run: fit current ratings, predict upcoming games, apply injuries, compare to lines, write picks.json + index.html data."""
import json, re, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from math import erf, sqrt
from walk import *
from players import load_players, REPL, PRIOR_MIN
SRC = "data/raw/mbb" if len(sys.argv) < 2 else sys.argv[1]
DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
p = json.load(open("data/params.json"))
cdf = lambda z: 0.5 * (1 + erf(z / sqrt(2)))

asof = pd.Timestamp(G.date.max()) + pd.Timedelta(days=1)
season = int(G.season.max())
_, prev, _ = run_chain(p, seasons=range(2020, season), step=14, keep_ratings=True)
prior = make_prior(prev, ret_at(season, asof), p, T)
rt = fit_ratings(OBS, None, asof, prior, p, T, p["hl"], p["lam"], POSS_DF)

sch = pd.read_parquet(f"{SRC}/schedules/parquet/mbb_schedule_{season}.parquet")
sch["game_date"] = pd.to_datetime(sch.game_date)
start = pd.Timestamp.now().normalize() if len(sys.argv) <= 3 else pd.Timestamp(sys.argv[3])
up = sch[(~sch.status_type_completed.astype(bool)) & (sch.game_date >= start) & (sch.game_date < start + pd.Timedelta(days=DAYS))].copy()
up["home_id"] = pd.to_numeric(up.home_id, errors="coerce"); up["away_id"] = pd.to_numeric(up.away_id, errors="coerce")
up = up[up.home_id.isin(IDX) & up.away_id.isin(IDX)]
up["hi"] = up.home_id.map(IDX).astype(int); up["ai"] = up.away_id.map(IDX).astype(int)
neu = up.neutral_site.fillna(False).astype(bool).values
pr = predict(rt, up.hi.values, up.ai.values, neu, T, p)

# ---- player layer: rotation + auto-detected absences + manual injuries.csv
P = load_players(); P = P[P.season == season]
pt = P.groupby(["team_id", "athlete_id", "athlete_display_name"]).agg(m=("minutes", "sum"), g=("gs", "sum"), n=("minutes", lambda x: (x > 0).sum())).reset_index()
pt["mpg"] = pt.m / pt.n.clip(lower=1); pt["v40"] = (pt.g + PRIOR_MIN / 40 * REPL) / ((pt.m + PRIOR_MIN) / 40)
pt["val"] = pt.mpg / 40 * (pt.v40 - REPL)
last = P.sort_values("game_date").groupby("team_id").game_date.max().rename("last")
played_last = P[P.minutes > 0].merge(last, on="team_id"); played_last = set(map(tuple, played_last[played_last.game_date == played_last["last"]][["team_id", "athlete_id"]].values))
pt["auto_out"] = [(t, a) not in played_last and m >= 14 and n >= 2 for t, a, m, n in zip(pt.team_id, pt.athlete_id, pt.mpg, pt.n)]
inj = pd.read_csv("data/injuries.csv") if __import__("os").path.exists("data/injuries.csv") else pd.DataFrame(columns=["team", "player", "status"])
PO = {"out": 1.0, "doubtful": 0.8, "questionable": 0.5, "day-to-day": 0.4, "probable": 0.15, "available": 0.0}
inj["p"] = inj.status.str.lower().map(PO).fillna(0.5)
def team_miss(tid, tname):
    x = pt[(pt.team_id == tid) & (pt.mpg >= 8)].copy()
    x["p_out"] = np.where(x.auto_out, 0.5, 0.0)
    for _, r in inj[inj.team.str.lower() == str(tname).lower()].iterrows():
        k = x.athlete_display_name.str.lower() == str(r.player).lower()
        x.loc[k, "p_out"] = r.p
    x = x[x.p_out > 0]
    return float((x.p_out * x.val).sum()), [f"{r.athlete_display_name} ({r.mpg:.0f} mpg, {'listed' if (inj.player.str.lower() == r.athlete_display_name.lower()).any() else 'auto'} {int(r.p_out*100)}%)" for r in x.sort_values("val", ascending=False).itertuples()]
rows = []
lines = pd.read_csv("data/lines.csv", dtype={"game_id": str}) if __import__("os").path.exists("data/lines.csv") else pd.DataFrame(columns=["game_id", "spread_home", "total"])
lines = lines.set_index("game_id")
for k, r in enumerate(up.itertuples()):
    mh, nh = team_miss(int(r.home_id), r.home_display_name); ma, na = team_miss(int(r.away_id), r.away_display_name)
    ph = pr["pts_h"][k] + p["tot_bias"] / 2 + p["margin_bias"] / 2 - p["k_off"] * mh + p["k_opp"] * ma
    pa = pr["pts_a"][k] + p["tot_bias"] / 2 - p["margin_bias"] / 2 - p["k_off"] * ma + p["k_opp"] * mh
    if neu[k]: ph, pa = ph - p["margin_bias"] / 2 * 0, pa
    gid = str(r.game_id); ln = lines.loc[gid] if gid in lines.index else None
    rows.append(dict(game_id=gid, date=str(r.game_date.date()), home=r.home_display_name, away=r.away_display_name, neutral=bool(neu[k]),
                     pts_h=round(float(ph), 1), pts_a=round(float(pa), 1), poss=round(float(pr["poss"][k]), 1),
                     model_spread=round(float(pa - ph), 1), model_total=round(float(ph + pa), 1),
                     win_prob_home=round(cdf((ph - pa) / p["sigma_margin"]), 3),
                     miss_home=round(mh, 2), miss_away=round(ma, 2), out_home=nh, out_away=na,
                     line_spread=None if ln is None else float(ln.spread_home), line_total=None if ln is None or pd.isna(ln.total) else float(ln.total)))
out = dict(generated=str(pd.Timestamp.now()), data_through=str(G.date.max().date()), sigma_margin=p["sigma_margin"], sigma_total=p["sigma_total"], games=rows)
json.dump(out, open("picks.json", "w"), indent=1)
if __import__("os").path.exists("index.html"):
    h = open("index.html").read()
    h = re.sub(r"/\*DATA\*/.*?/\*END\*/", lambda m: "/*DATA*/" + json.dumps(out) + "/*END*/", h, flags=re.S); open("index.html", "w").write(h)
log = pd.DataFrame(rows).assign(run=str(pd.Timestamp.now())).drop(columns=["out_home", "out_away"])
log.to_csv("data/predictions_log.csv", mode="a", header=not __import__("os").path.exists("data/predictions_log.csv"), index=False)
print(len(rows), "games ->", "picks.json")
