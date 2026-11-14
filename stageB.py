import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from walk import *
from tune import score
from scipy.optimize import minimize
best = json.load(open("data/params_stageA.json"))
obj = lambda r: r[0]/12 + r[1]/17.5
cur = score(run_chain(best, seasons=range(2020,2027)), best); print("A", cur, flush=True)
for k, v in [("c_base", 0.95), ("c_ret", 2.0), ("hl", 200)]:
    q = dict(best); q[k] = v; r = score(run_chain(q, seasons=range(2020,2027)), q); print(k, v, r, flush=True)
    if obj(r) < obj(cur) - 1e-6: best, cur = q, r
print("stageA final", best, cur, flush=True)
d, prev, store = run_chain(best, seasons=range(2020,2027), keep_ratings=True)
ev = d[d.season >= 2023].reset_index(drop=True)
gi = G.set_index("game_id")
def repred(post):
    q = dict(best); q.update(post); parts = []
    for (s, c), grp in ev.groupby(["season", "cut"]):
        pr = predict(store[(s, pd.Timestamp(c))], grp.hi.values, grp.ai.values, grp.neutral.values, T, q)
        parts.append(pd.DataFrame({"pts_h": pr["pts_h"], "pts_a": pr["pts_a"], "poss_pred": pr["poss"]}, index=grp.index))
    return pd.concat(parts).sort_index()
def loss(x):
    post = dict(s=x[0], s_pace=x[1], s_ftp=x[2], ppp_scale=x[3])
    r = repred(post); tot = r.pts_h + r.pts_a; mar = r.pts_h - r.pts_a
    ta = ev.act_h + ev.act_a; ma = ev.act_h - ev.act_a
    return np.sqrt(((mar - ma - (mar - ma).mean())**2).mean())/12 + np.sqrt(((tot - ta - (tot - ta).mean())**2).mean())/17.5
x0 = [1.0, 1.0, 0.6, 1.0]
res = minimize(loss, x0, method="Nelder-Mead", options=dict(maxiter=60, xatol=0.01, fatol=1e-5))
print("stageB", res.x, res.fun, "start", loss(x0), flush=True)
post = dict(s=float(res.x[0]), s_pace=float(res.x[1]), s_ftp=float(res.x[2]), ppp_scale=float(res.x[3]))
best.update(post); r = repred(post)
best["tot_bias"] = float((ev.act_h + ev.act_a - r.pts_h - r.pts_a).mean()); best["margin_bias"] = float((ev.act_h - ev.act_a - r.pts_h + r.pts_a).mean())
ev = ev.assign(pts_h=r.pts_h, pts_a=r.pts_a, poss_pred=r.poss_pred)
ev.to_parquet("data/wf_final.parquet")
json.dump(best, open("data/params.json", "w"), indent=1); print("FINAL", best)
print(metrics(ev, best, lo_season=2023))
