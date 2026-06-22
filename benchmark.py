# benchmark.py
import os
import numpy as np
import torch
from statistics import median

PHASE1_STEPS = 20_000
TOPK_PER_GROUP = 2

USE_BENCHMARK_CACHE = True


def td_slope(td_log, frac=0.6):
    if len(td_log) < 5:
        return 0.0
    s, v = zip(*td_log)
    s = np.asarray(s, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    k = int(len(s) * frac)
    s, v = s[-k:], v[-k:]
    s -= s.mean()
    v -= v.mean()
    denom = (s ** 2).sum()
    return 0.0 if denom < 1e-8 else float((s * v).sum() / denom)


def td_resid_std(td_log):
    if len(td_log) < 5:
        return float("inf")
    s, v = zip(*td_log)
    s = np.asarray(s, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    a = td_slope(td_log)
    b = v.mean() - a * s.mean()
    return float((v - (a * s + b)).std())


def norm(xs):
    lo, hi = min(xs), max(xs)
    return [0.0 if hi <= lo else (x - lo) / (hi - lo) for x in xs]


def load_cached(ckpt_path):
    try:
        ck = torch.load(ckpt_path, map_location="cpu")
        return {
            "td_log": ck["td_log"],
            "eval_p50": ck["eval_p50"],
            "it_per_s": ck["it_per_s"],
        }
    except Exception:
        return None


def benchmark_training_grid(
    model_type, grid,
    train_batches, train_envs,
    repeats, short_steps,
    bench_dir,
    train_mlp, train_cnn, make_eval_env,
    load_ckpt=None,
):
    os.makedirs(bench_dir, exist_ok=True)
    eval_env = make_eval_env(grid)

    print(f"\n=== PHASE 1 SAFE BENCH g{grid} steps={PHASE1_STEPS} ===")
    rows = []

    for B in train_batches:
        for E in train_envs:
            slopes, rstds, p50s, its = [], [], [], []
            print(f"\nConfig B={B} E={E}")

            for r in range(repeats):
                ckpt = os.path.join(
                    bench_dir, f"p1_{model_type}_g{grid}_B{B}_E{E}_r{r}.pt"
                )

                if USE_BENCHMARK_CACHE and os.path.exists(ckpt):
                    print(f"  Using cached benchmark: {ckpt}")
                    cached = load_cached(ckpt)
                    if cached is not None:
                        m = cached
                        slopes.append(td_slope(m["td_log"]))
                        rstds.append(td_resid_std(m["td_log"]))
                        p50s.append(m["eval_p50"])
                        its.append(m["it_per_s"])
                        continue

                print(f"  Running benchmark fresh: {ckpt}")
                m = (train_mlp if model_type == "mlp" else train_cnn)(
                    grid, PHASE1_STEPS, ckpt, B, E,
                    load_ckpt=load_ckpt,
                    eval_env=eval_env,
                )

                raw = torch.load(ckpt, map_location="cpu")
                raw["td_log"] = m["td_log"]
                raw["eval_p50"] = m["eval_p50"]
                raw["it_per_s"] = m["it_per_s"]
                torch.save(raw, ckpt)

                slopes.append(td_slope(m["td_log"]))
                rstds.append(td_resid_std(m["td_log"]))
                p50s.append(m["eval_p50"])
                its.append(m["it_per_s"])

            rows.append({
                "batch": B,
                "n_envs": E,
                "td_slope": median(slopes),
                "td_resid_std": median(rstds),
                "eval_p50": median(p50s),
                "it_med": median(its),
                "best_ckpt": ckpt,
            })

    ns = {
        "s": norm([r["td_slope"] for r in rows]),
        "r": norm([r["td_resid_std"] for r in rows]),
        "p": norm([r["eval_p50"] for r in rows]),
        "i": norm([r["it_med"] for r in rows]),
    }

    for r, s, rs, p, it in zip(
        rows, ns["s"], ns["r"], ns["p"], ns["i"]
    ):
        r["score_p1"] = -0.40 * s - 0.25 * rs + 0.25 * p + 0.10 * it

    shortlist = []
    for B in train_batches:
        grp = [r for r in rows if r["batch"] == B]
        grp.sort(key=lambda r: r["score_p1"], reverse=True)
        shortlist += grp[:TOPK_PER_GROUP]

    for E in train_envs:
        grp = [r for r in rows if r["n_envs"] == E]
        grp.sort(key=lambda r: r["score_p1"], reverse=True)
        shortlist += grp[:TOPK_PER_GROUP]

    rows.sort(key=lambda r: r["score_p1"], reverse=True)
    shortlist += rows[:2]

    shortlist = {(r["batch"], r["n_envs"]): r for r in shortlist}.values()
    shortlist = list(shortlist)

    print(f"\nPhase‑1 survivors: {len(shortlist)}")
    return shortlist
