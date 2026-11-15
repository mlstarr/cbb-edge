"""Grade logged predictions against final scores -> results.json and the Results tab of index.html.
Sections: most recent graded day, season-to-date vs the market (from logged lines), model-only accuracy (walk-forward, no lines),
and a tracker for the parlay tickets the dashboard would have suggested (flat stakes)."""
import json, os, re, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from math import exp, log
SPREAD_EDGE, TOTAL_EDGE = 3.0, 4.0
WIN = 100 / 110                       # profit per 1u at -110
P = json.load(open("data/params.json"))

def grade_side(diff):                 # diff>0 win, <0 loss, 0 push
    return "W" if diff > 0 else "L" if diff < 0 else "P"

def load_graded(logfile="data/predictions_log.csv", gamesfile="data/games.parquet"):
    if not os.path.exists(logfile): return pd.DataFrame()
    L = pd.read_csv(logfile, dtype={"game_id": str}); L["run"] = pd.to_datetime(L.run)
    L = L.sort_values("run")
    has = L.dropna(subset=["line_spread"]).groupby("game_id").tail(1)          # last snapshot that had a line
    last = L.groupby("game_id").tail(1)
    L = pd.concat([has, last[~last.game_id.isin(has.game_id)]])
    g = pd.read_parquet(gamesfile)[["game_id", "team_score_h", "team_score_a"]]; g["game_id"] = g.game_id.astype(str)
    d = L.merge(g, on="game_id", how="inner").rename(columns={"team_score_h": "act_h", "team_score_a": "act_a"})
    d["model_margin"] = d.pts_h - d.pts_a; d["act_margin"] = d.act_h - d.act_a; d["act_total"] = d.act_h + d.act_a
    d["err_margin"] = d.model_margin - d.act_margin; d["err_total"] = d.model_total - d.act_total
    d["se"] = d.model_margin + d.line_spread; d["te"] = d.model_total - d.line_total
    cov = d.act_margin + d.line_spread                                       # >0 home covers
    d["ats"] = [("" if pd.isna(s) else grade_side(c if s > 0 else -c)) for s, c in zip(d.se, cov)]
    d["tot"] = [("" if pd.isna(t) else grade_side((a - l) if t > 0 else (l - a))) for t, a, l in zip(d.te, d.act_total, d.line_total)]
    return d

def rec(res, mask):
    r = res[mask]; w, l, p = (r == "W").sum(), (r == "L").sum(), (r == "P").sum()
    return dict(w=int(w), l=int(l), p=int(p), units=round(float(w * WIN - l), 2), pct=round(float(w / (w + l)), 3) if w + l else None)

def plays(d):
    out = []
    for r in d.sort_values("date").itertuples():
        base = dict(date=str(r.date), game=f"{r.away} @ {r.home}", final=f"{int(r.act_a)}-{int(r.act_h)}")
        if r.ats != "" and abs(r.se) >= SPREAD_EDGE:
            h = r.se > 0; out.append(dict(base, kind="Spread", pick=f"{r.home if h else r.away} {'%+g' % (r.line_spread if h else -r.line_spread)}", edge=round(abs(r.se), 1), result=r.ats, units=round(WIN if r.ats == "W" else -1.0 if r.ats == "L" else 0.0, 2)))
        if r.tot != "" and abs(r.te) >= TOTAL_EDGE:
            out.append(dict(base, kind="Total", pick=f"{'Over' if r.te > 0 else 'Under'} {r.line_total:g}", edge=round(abs(r.te), 1), result=r.tot, units=round(WIN if r.tot == "W" else -1.0 if r.tot == "L" else 0.0, 2)))
    return sorted(out, key=lambda x: -x["edge"])

def season_summary(d):
    o = {}
    if d.empty: return o
    o["games"] = int(len(d)); o["with_lines"] = int(d.line_spread.notna().sum())
    o["spread_all"] = rec(d.ats, d.ats != ""); o["spread_edge"] = rec(d.ats, (d.se.abs() >= SPREAD_EDGE) & (d.ats != ""))
    o["total_all"] = rec(d.tot, d.tot != ""); o["total_edge"] = rec(d.tot, (d.te.abs() >= TOTAL_EDGE) & (d.tot != ""))
    b = []
    for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 10), (10, 99)]:
        m = (d.se.abs() >= lo) & (d.se.abs() < hi) & (d.ats != ""); t = (d.te.abs() >= lo) & (d.te.abs() < hi) & (d.tot != "")
        b.append(dict(bucket=f"{lo}-{hi if hi < 99 else '+'}", spread=rec(d.ats, m), total=rec(d.tot, t)))
    o["buckets"] = b
    dd = d.sort_values("date"); cs = []; us = ut = 0.0
    for day, x in dd.groupby("date"):
        a = x[(x.se.abs() >= SPREAD_EDGE) & (x.ats != "")].ats; t = x[(x.te.abs() >= TOTAL_EDGE) & (x.tot != "")].tot
        us += (a == "W").sum() * WIN - (a == "L").sum(); ut += (t == "W").sum() * WIN - (t == "L").sum()
        cs.append(dict(date=str(day), spread=round(float(us), 2), total=round(float(ut), 2)))
    o["curve"] = cs
    o["mae_margin"] = round(float(d.err_margin.abs().mean()), 2); o["mae_total"] = round(float(d.err_total.abs().mean()), 2)
    o["bias_margin"] = round(float(d.err_margin.mean()), 2); o["bias_total"] = round(float(d.err_total.mean()), 2)
    o["su"] = round(float(((d.model_margin > 0) == (d.act_margin > 0)).mean()), 3)
    return o

def legs_for_day(x, sh=0.35, cap=0.57, mx=10.0, sm=11.79, st=17.10):
    from math import erf, sqrt
    cdf = lambda z: 0.5 * (1 + erf(z / sqrt(2))); out = []
    for r in x.itertuples():
        c = []
        if pd.notna(r.line_spread) and r.se != 0 and abs(r.se) <= mx: c.append((min(cap, .5 + sh * (cdf(abs(r.se) / sm) - .5)), f"{r.home if r.se > 0 else r.away} {'%+g' % (r.line_spread if r.se > 0 else -r.line_spread)}", r.ats))
        if pd.notna(r.line_total) and r.te != 0 and abs(r.te) <= mx: c.append((min(cap, .5 + sh * (cdf(abs(r.te) / st) - .5)), f"{'Over' if r.te > 0 else 'Under'} {r.line_total:g} ({r.away.split()[0]}/{r.home.split()[0]})", r.tot))
        if c: out.append(max(c))
    return sorted(out, key=lambda z: -z[0])

def parlay_tracker(d, am=-110, small_u=1.0, lotto_u=0.25):
    dec = 1 + 100 / -am; days = []; tot = dict(small=0.0, lotto=0.0); n = dict(small=0, lotto=0); h = dict(small=0, lotto=0)
    for day, x in d.groupby("date"):
        L = legs_for_day(x)
        if len(L) < 2: continue
        best = None
        for k in (2, 3, 4):
            if k > len(L): break
            pp = np.prod([l[0] for l in L[:k]]); D = dec ** k; b = D - 1; f = (pp * b - (1 - pp)) / b
            gr = pp * log(1 + f * b) + (1 - pp) * log(1 - f) if f > 0 else 0
            if f > 0 and (best is None or gr > best[0]): best = (gr, k)
        row = dict(date=str(day))
        for name, k, u in [("small", best[1] if best else None, small_u), ("lotto", min(8, len(L)), lotto_u)]:
            if not k or (name == "lotto" and len(L) < 6): continue
            legs = L[:k]; res = [l[2] for l in legs]
            if "L" in res: pnl = -u; out = "L"
            else:
                live = sum(r == "W" for r in res); pnl = u * (dec ** live - 1) if live else 0.0; out = "W" if live else "P"
            tot[name] += pnl; n[name] += 1; h[name] += out == "W"
            row[name] = dict(k=k, result=out, pnl=round(pnl, 2), legs=[l[1] + " " + l[2] for l in legs])
        days.append(row)
    return dict(days=days[-14:], small=dict(n=n["small"], hit=h["small"], units=round(tot["small"], 2)), lotto=dict(n=n["lotto"], hit=h["lotto"], units=round(tot["lotto"], 2)),
                note="Flat stakes: small parlay 1u, lotto 0.25u, -110 legs, one leg per game, last logged line before the game.")

def model_only(season):
    """walk-forward predictions for this season's finished games (no lines needed)."""
    from walk import run_chain, G
    p = dict(P); d = run_chain(p, seasons=range(2020, season + 1), step=7)
    d = d[d.season == season].copy()
    from model import early_factor
    ef = early_factor((d.date - d.date.min()).dt.days.values, p); eh = np.where(d.neutral.values, 0.0, p["early_margin"]) * ef
    d["em"] = d.pts_h - d.pts_a + p["margin_bias"] + eh - (d.act_h - d.act_a); d["et"] = d.pts_h + d.pts_a + p["tot_bias"] + p["early_total"] * ef - (d.act_h + d.act_a)
    d["su"] = ((d.pts_h - d.pts_a + p["margin_bias"] + eh) > 0) == ((d.act_h - d.act_a) > 0)
    last7 = d[d.date >= d.date.max() - pd.Timedelta(days=6)]
    f = lambda x: dict(n=int(len(x)), mae_margin=round(float(x.em.abs().mean()), 2), mae_total=round(float(x.et.abs().mean()), 2), bias_margin=round(float(x.em.mean()), 2), bias_total=round(float(x.et.mean()), 2), su=round(float(x.su.mean()), 3))
    return dict(season=f(d), last7=f(last7), through=str(d.date.max().date()), note="Walk-forward on finished games using only data before each game; no betting lines involved.")

def main():
    d = load_graded()
    out = dict(generated=str(pd.Timestamp.now())[:16])
    if d.empty:
        out["yesterday"] = None; out["season"] = {}; out["parlays"] = None
    else:
        last_day = d.date.max(); y = d[d.date == last_day].sort_values("date")
        out["yesterday"] = dict(date=str(last_day), summary=season_summary(y), plays=plays(y), games=[dict(away=r.away, home=r.home, final=f"{int(r.act_a)}-{int(r.act_h)}", model_margin=round(r.model_margin, 1), model_total=round(r.model_total, 1), line_spread=None if pd.isna(r.line_spread) else r.line_spread, line_total=None if pd.isna(r.line_total) else r.line_total, ats=r.ats, tot=r.tot, err_margin=round(r.err_margin, 1), err_total=round(r.err_total, 1)) for r in y.itertuples()])
        out["season"] = season_summary(d); out["parlays"] = parlay_tracker(d)
    try: out["model_only"] = model_only(int(pd.read_parquet("data/games.parquet", columns=["season"]).season.max()))
    except Exception as e: out["model_only"] = None; print("model-only skipped:", e)
    json.dump(out, open("results.json", "w"), indent=1, default=str)
    if os.path.exists("index.html"):
        h = open("index.html").read()
        h = re.sub(r"/\*RES\*/.*?/\*RESEND\*/", lambda m: "/*RES*/" + json.dumps(out, default=str) + "/*RESEND*/", h, flags=re.S); open("index.html", "w").write(h)
    print("results ->", "graded games:", 0 if d.empty else len(d))
if __name__ == "__main__": main()
