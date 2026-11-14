import numpy as np, pandas as pd, time, pickle, sys
from model import *

G, IDS, IDX = load(); T = len(IDS)
OBS = build_obs(G)
POSS_DF = (G.hi.values, G.ai.values, G.date.values)
PL = pd.read_parquet("data/players.parquet", columns=["season","game_id","game_date","athlete_id","team_id","minutes"])
PL["ti"] = PL.team_id.map(IDX)
_first = PL[PL.minutes > 0].groupby(["season","ti","athlete_id"]).game_date.min().rename("first").reset_index()
_min = PL.groupby(["season","ti","athlete_id"]).minutes.sum().rename("min").reset_index()
_min["season"] += 1                                    # last season's minutes, keyed to the season they carry into
_tot = _min.groupby(["season","ti"])["min"].sum().rename("tot").reset_index()
_RET = _min.merge(_first, on=["season","ti","athlete_id"], how="inner")
_GAMES = pd.concat([G[["season","date","hi"]].rename(columns={"hi":"ti"}), G[["season","date","ai"]].rename(columns={"ai":"ti"})])

def ret_at(season, cutoff):
    r = np.full(T, np.nan)
    sub = _RET[(_RET.season == season) & (_RET["first"] < cutoff)]
    tot = _tot[_tot.season == season].set_index("ti")["tot"]
    got = sub.groupby("ti")["min"].sum()
    gm = _GAMES[(_GAMES.season == season) & (_GAMES.date < cutoff)].groupby("ti").size()
    for t, v in got.items():
        w = min(1.0, gm.get(t, 0) / 4.0)
        r[t] = w * v / tot.get(t, np.inf) + (1 - w) * 0.5
    return r

def season_cutoffs(season, step):
    gs = G[G.season == season].date
    start, end = gs.min(), gs.max()
    cuts = list(pd.date_range(start, end + pd.Timedelta(days=1), freq=f"{step}D"))
    return start, end, cuts

def run_chain(p, seasons=range(2018, 2027), step=14, verbose=False, keep_ratings=False):
    """walk-forward: returns per-game predictions (raw model output) and the per-cutoff ratings."""
    prev = None; rows = []; store = {}
    for s in seasons:
        ret = ret_at(s, pd.Timestamp(G[G.season == s].date.min())) if prev is not None else None
        start, end, cuts = season_cutoffs(s, step)
        prior_ret = lambda c: ret_at(s, c) if prev is not None else None
        pr0 = make_prior(prev, ret_at(s, start + pd.Timedelta(days=-1)) if prev is not None else None, p, T)
        gs = G[G.season == s]
        for k, c in enumerate(cuts):
            rt_prior = make_prior(prev, ret_at(s, c) if prev is not None else None, p, T)
            rt = fit_ratings(OBS, None, c, rt_prior, p, T, p["hl"], p["lam"], POSS_DF)
            nxt = cuts[k + 1] if k + 1 < len(cuts) else end + pd.Timedelta(days=2)
            win = gs[(gs.date >= c) & (gs.date < nxt)]
            if len(win):
                pr = predict(rt, win.hi.values, win.ai.values, win.neutral_site.values, T, p)
                d = pd.DataFrame({"game_id": win.game_id.values, "season": s, "date": win.date.values, "hi": win.hi.values, "ai": win.ai.values,
                                  "neutral": win.neutral_site.values, "poss_pred": pr["poss"], "ppp_h": pr["ppp_h"], "ppp_a": pr["ppp_a"],
                                  "pts_h": pr["pts_h"], "pts_a": pr["pts_a"],
                                  "act_h": win.team_score_h.values, "act_a": win.team_score_a.values, "act_poss": win.poss.values, "cut": c})
                rows.append(d)
            if keep_ratings: store[(s, c)] = rt
        # final fit for next season's prior
        endc = end + pd.Timedelta(days=2)
        fpr = make_prior(prev, ret_at(s, endc) if prev is not None else None, p, T)
        prev = fit_ratings(OBS, None, endc, fpr, p, T, p["hl_final"], p["lam_final"], POSS_DF)
        if verbose: print("season", s, "done", flush=True)
    out = pd.concat(rows, ignore_index=True)
    return (out, prev, store) if keep_ratings else out

def metrics(d, p, lo_season=2022, skip_days=0):
    d = d[(d.season >= lo_season) & (d.season != 2021)]
    st = d.groupby("season").date.transform("min")
    d = d[(d.date - st).dt.days >= skip_days]
    tot_p = d.pts_h + d.pts_a + p["tot_bias"]; mar_p = d.pts_h - d.pts_a + p["margin_bias"]
    tot_a = d.act_h + d.act_a; mar_a = d.act_h - d.act_a
    return dict(n=len(d), rmse_margin=np.sqrt(((mar_p - mar_a) ** 2).mean()), mae_margin=np.abs(mar_p - mar_a).mean(),
                rmse_total=np.sqrt(((tot_p - tot_a) ** 2).mean()), mae_total=np.abs(tot_p - tot_a).mean(),
                bias_total=(tot_p - tot_a).mean(), bias_margin=(mar_p - mar_a).mean(),
                rmse_poss=np.sqrt(((d.poss_pred - d.act_poss) ** 2).mean()),
                win_acc=((mar_p > 0) == (mar_a > 0)).mean())

if __name__ == "__main__":
    t0 = time.time()
    p = dict(DEFAULT)
    d = run_chain(p, verbose=True)
    print("time", round(time.time() - t0, 1))
    d.to_parquet("data/wf_default.parquet")
    print(metrics(d, p))
    # naive baselines
    dd = d[(d.season >= 2022) & (d.season != 2021)]
    ma = dd.act_h - dd.act_a; ta = dd.act_h + dd.act_a
    print("naive margin rmse (home +%.1f)" % ma.mean(), np.sqrt(((ma - ma.mean()) ** 2).mean()), " naive total rmse", ta.std())
