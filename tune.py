"""Stage A: tune rating-fit hyper-parameters walk-forward (objective = margin RMSE + total RMSE on 2022-26).
Stage B: tune cheap post-hoc parameters (scale, exponent, bias) on the cached predictions."""
import numpy as np, pandas as pd, json, time, itertools, warnings
warnings.filterwarnings("ignore")
from walk import *
from scipy.optimize import minimize

def score(d, p, skip=0):
    # calibrate post-hoc scale/bias inside score so stage A is judged fairly
    dd = d[(d.season >= 2023)]
    st = dd.groupby("season").date.transform("min"); dd = dd[(dd.date - st).dt.days >= skip]
    tot_p = dd.pts_h + dd.pts_a; mar_p = dd.pts_h - dd.pts_a
    tot_a = dd.act_h + dd.act_a; mar_a = dd.act_h - dd.act_a
    tb = (tot_a - tot_p).mean()
    mb = (mar_a - mar_p).mean()
    r_t = np.sqrt(((tot_p + tb - tot_a) ** 2).mean()); r_m = np.sqrt(((mar_p + mb - mar_a) ** 2).mean())
    return r_m, r_t

if __name__ == "__main__":
    t0 = time.time()
    base = dict(DEFAULT)
    grids = [("hl", [30, 60, 110, 400]), ("lam", [8, 22, 40]), ("c_base", [0.5, 0.72, 0.85]), ("c_ret", [0.0, 0.9, 1.4])]
    best = dict(base); br = score(run_chain(best, seasons=range(2020,2027), step=14), best); print("start", br, flush=True)
    obj = lambda r: r[0] / 12.0 + r[1] / 17.5
    bo = obj(br)
    for sweep in range(1):                      # coordinate descent, single pass
        for k, vals in grids:
            res = []
            for v in vals:
                q = dict(best); q[k] = v
                r = score(run_chain(q, seasons=range(2020,2027), step=14), q); res.append((obj(r), v, r))
            res.sort(key=lambda x: x[0])
            if res[0][0] < bo - 1e-6:
                bo = res[0][0]; best[k] = res[0][1]; br = res[0][2]
            print(f"sweep{sweep} {k}: best={best[k]} margin={br[0]:.3f} total={br[1]:.3f}  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(best, open("data/params_stageA.json", "w"), indent=1)
    print("BEST", best, br)
