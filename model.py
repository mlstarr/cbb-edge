"""College basketball predictor core.

Odds-ratio (log5-style) model on the four factors + FT%:
    logit(rate_game) = mu + s * (offense_effect[team] + defense_effect[opp]) + home_effect
i.e. each team's rate is compared to the D1 average on the odds scale and the two odds ratios are combined
(Bill James / Morey odds-ratio method). Effects are fit jointly (opponent-adjusted) by ridge-penalised
IRLS with recency weights, shrunk toward a prior built from last season's ratings and returning minutes.
Pace is an additive tempo model. Possessions x points-per-possession gives scores, spread and total.
"""
import numpy as np, pandas as pd, scipy.sparse as sp

FACTORS = ["efg", "tov", "orb", "ftr", "ftp"]
FAMILY = {"efg": "logit", "tov": "logit", "orb": "logit", "ftr": "log", "ftp": "logit"}
USE_DEF = {"efg": True, "tov": True, "orb": True, "ftr": True, "ftp": False}
KAPPA = 1.13   # eFG% / FG% ratio (used only to turn eFG into a miss rate for ORB opportunities)

DEFAULT = dict(hl=45.0, lam=14.0, lam_g=300.0, c_base=0.62, c_ret=0.55, hl_final=120.0, lam_final=6.0,
               s=1.0, s_pace=1.0, s_ftp=0.6, tot_bias=0.0, margin_bias=0.0, ppp_scale=1.0, hca_scale=1.0)

def load():
    g = pd.read_parquet("data/games.parquet")
    g["date"] = pd.to_datetime(g.date)
    ids = np.unique(np.concatenate([g.hid.values, g.aid.values]))
    idx = {t: i for i, t in enumerate(ids)}
    g["hi"] = g.hid.map(idx); g["ai"] = g.aid.map(idx)
    return g, ids, idx

def build_obs(g):
    """two observation rows per game (home offense vs away defense, away offense vs home defense)."""
    n = len(g)
    ns = (~g.neutral_site.values).astype(float)
    out = {}
    out["off"] = np.concatenate([g.hi.values, g.ai.values]); out["def"] = np.concatenate([g.ai.values, g.hi.values])
    out["hs"] = np.concatenate([ns, -ns])
    out["date"] = np.concatenate([g.date.values, g.date.values]); out["row"] = np.concatenate([np.arange(n), np.arange(n)])
    def side(x, o):
        return dict(fgm=g[f"field_goals_made_{x}"].values, fga=g[f"field_goals_attempted_{x}"].values,
                    tpm=g[f"three_point_field_goals_made_{x}"].values, fta=g[f"free_throws_attempted_{x}"].values,
                    ftm=g[f"free_throws_made_{x}"].values, orb=g[f"offensive_rebounds_{x}"].values,
                    drb_o=g[f"defensive_rebounds_{o}"].values, tov=g[f"tov_{x}"].values, poss=g.poss.values)
    H, A = side("h", "a"), side("a", "h")
    cat = lambda k: np.concatenate([H[k], A[k]]).astype(float)
    fga, fgm, tpm, fta, ftm = cat("fga"), cat("fgm"), cat("tpm"), cat("fta"), cat("ftm")
    orb, drb, tov, poss = cat("orb"), cat("drb_o"), cat("tov"), cat("poss")
    out["y"] = {"efg": (fgm + 0.5 * tpm) / fga, "tov": tov / poss, "orb": orb / (orb + drb), "ftr": fta / fga, "ftp": ftm / np.maximum(fta, 1)}
    out["n"] = {"efg": fga, "tov": poss, "orb": orb + drb, "ftr": fga, "ftp": fta}
    out["poss"] = g.poss.values
    return out

def _design(off, dfn, hs, T, use_def):
    N = len(off); r = np.arange(N)
    rows = [r, r, r]; cols = [off, np.full(N, 2 * T), np.full(N, 2 * T + 1)]; vals = [np.ones(N), np.ones(N), hs]
    if use_def:
        rows.append(r); cols.append(T + dfn); vals.append(np.ones(N))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, 2 * T + 2))

def irls(X, y, n, rw, fam, beta0, pen, iters=4):
    beta = beta0.copy()
    sc = None
    for _ in range(iters):
        eta = X @ beta
        if fam == "logit":
            mu = 1 / (1 + np.exp(-np.clip(eta, -8, 8))); v = np.clip(mu * (1 - mu), 1e-4, None); dmu = v
        else:
            mu = np.exp(np.clip(eta, -8, 5)); v = mu; dmu = mu
        z = eta + (y - mu) / np.clip(dmu, 1e-6, None)
        w = rw * n * v
        if sc is None: sc = w.mean()
        w = w / sc
        XtW = X.T.multiply(w).tocsr()
        A = (XtW @ X).toarray(); A[np.diag_indices_from(A)] += pen
        b = XtW @ z + pen * beta0
        beta = np.linalg.solve(A, b)
    return beta

def fit_ratings(obs, mask_games, cutoff, prior, p, T, hl, lam, poss_df):
    """Fit all factor models + pace using obs rows with date<cutoff. prior: dict factor->beta0 (len 2T+2), 'pace'->(len T+1)."""
    N2 = len(obs["row"])
    m = obs["date"] < np.datetime64(cutoff)
    out = {}
    if m.sum() < 50:
        return {k: v.copy() for k, v in prior.items()}
    age = (np.datetime64(cutoff) - obs["date"][m]).astype("timedelta64[D]").astype(float)
    rw = 0.5 ** (age / hl)
    for f in FACTORS:
        X = _design(obs["off"][m], obs["def"][m], obs["hs"][m], T, USE_DEF[f])
        pen = np.full(2 * T + 2, lam); pen[2 * T:] = p["lam_g"]
        if not USE_DEF[f]: pen[T:2 * T] = 1e9
        out[f] = irls(X, obs["y"][f][m], obs["n"][f][m], rw, FAMILY[f], prior[f], pen)
    # pace: poss = mu + t_h + t_a
    gm = np.zeros(len(obs["poss"]), bool); gm[obs["row"][m]] = True
    rows = np.where(gm)[0]
    ghi, gai, gdate = poss_df
    ageg = (np.datetime64(cutoff) - gdate[rows]).astype("timedelta64[D]").astype(float); wg = 0.5 ** (ageg / hl)
    r = np.arange(len(rows))
    X = sp.csr_matrix((np.ones(3 * len(rows)), (np.concatenate([r, r, r]), np.concatenate([ghi[rows], gai[rows], np.full(len(rows), T)]))), shape=(len(rows), T + 1))
    pen = np.full(T + 1, lam * 1.5); pen[T] = p["lam_g"]
    pb = prior["pace"]
    XtW = X.T.multiply(wg).tocsr(); A = (XtW @ X).toarray(); sc = wg.mean() if wg.mean() > 0 else 1
    A = A / sc; A[np.diag_indices_from(A)] += pen
    b = (XtW @ obs["poss"][rows]) / sc + pen * pb
    out["pace"] = np.linalg.solve(A, b)
    return out

def make_prior(prev, ret, p, T):
    """prior center from last season's final ratings; ret = returning-minutes share per team (nan -> mean)."""
    if prev is None:
        pr = {f: np.zeros(2 * T + 2) for f in FACTORS}; pr["pace"] = np.zeros(T + 1); pr["pace"][T] = 69.0
        pr["efg"][2 * T] = -0.06; pr["tov"][2 * T] = -1.7; pr["orb"][2 * T] = -0.85; pr["ftr"][2 * T] = -1.1; pr["ftp"][2 * T] = 0.8
        return pr
    rm = np.nanmean(ret) if np.isfinite(ret).any() else 0.5
    carry = np.clip(p["c_base"] + p["c_ret"] * (ret - rm), 0.15, 0.95)
    carry = np.where(np.isnan(carry), p["c_base"] * 0.8, carry)
    pr = {}
    for f in FACTORS:
        b = prev[f].copy()
        b[:T] = b[:T] * carry; b[T:2 * T] = b[T:2 * T] * carry
        pr[f] = b
    pb = prev["pace"].copy(); pb[:T] = pb[:T] * carry; pr["pace"] = pb
    return pr

def predict(rt, hi, ai, neutral, T, p):
    """vectorised; rt = ratings dict. returns dict with ppp, poss, pts_h, pts_a."""
    hs = (~neutral).astype(float)
    res = {}
    for who, (o, d, sg) in {"h": (hi, ai, hs), "a": (ai, hi, -hs)}.items():
        v = {}
        for f in FACTORS:
            b = rt[f]
            dd = b[T + d] if USE_DEF[f] else 0.0
            s = p["s_ftp"] if f == "ftp" else p["s"]
            eta = b[2 * T] + s * (b[o] + dd) + sg * b[2 * T + 1]
            v[f] = 1 / (1 + np.exp(-eta)) if FAMILY[f] == "logit" else np.exp(eta)
        fg = v["efg"] / KAPPA
        x = (1 - v["tov"]) / (1 - v["orb"] * (1 - fg) + 0.475 * v["ftr"])
        res["ppp_" + who] = p.get("ppp_scale", 1.0) * x * (2 * v["efg"] + v["ftr"] * v["ftp"])
    pb = rt["pace"]
    res["poss"] = pb[T] + p["s_pace"] * (pb[hi] + pb[ai])
    res["pts_h"] = res["poss"] * res["ppp_h"]; res["pts_a"] = res["poss"] * res["ppp_a"]
    return res

def returning_minutes(players, season, cutoff, T, idx):
    """share of last season's team minutes played by players who have appeared for the team this season before cutoff."""
    pl = players
    prev = pl[pl.season == season - 1].groupby(["team_id", "athlete_id"]).minutes.sum().reset_index()
    tot = prev.groupby("team_id").minutes.sum()
    cur = pl[(pl.season == season) & (pl.game_date < cutoff) & (pl.minutes > 0)].groupby(["team_id", "athlete_id"]).size().reset_index()[["team_id", "athlete_id"]]
    games = pl[(pl.season == season) & (pl.game_date < cutoff)].groupby("team_id").game_id.nunique()
    m = prev.merge(cur, on=["team_id", "athlete_id"])
    ret = m.groupby("team_id").minutes.sum() / tot
    r = np.full(T, np.nan)
    for t, v in ret.items():
        if t in idx and games.get(t, 0) > 0:
            w = min(1.0, games[t] / 4.0)
            r[idx[t]] = w * v + (1 - w) * 0.5
    return r
