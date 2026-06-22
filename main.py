import os
import argparse
import torch
import multiprocessing as mp

from benchmark import benchmark_training_grid
from trainer import (
    train_mlp,
    train_cnn,
    train_c51_cnn,
    run_ea_best_observed,
    make_eval_env,
)

_BASE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

BENCH_DIR = os.path.join(_BASE, "benchmarks")
CKPT_DIR = os.path.join(_BASE, "checkpoints")
MODEL_DIR = os.path.join(_BASE, "saved_models")

os.makedirs(BENCH_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DEFAULT_TRAIN_BATCHES = [512, 1024]
DEFAULT_TRAIN_ENVS = [256, 512, 1024]
DEFAULT_TRAIN_LRS = [2e-4, 3e-4, 5e-4]

C51_PHASE1_STEPS = 20_000
DEFAULT_C51_LRS = [2e-4, 3e-4]
DEFAULT_C51_ENVS = [256, 512, 1024]
DEFAULT_C51_ATOMS = [21, 51]
DEFAULT_C51_VRANGE = [(-10, 10), (-20, 20)]

FULL_TRAIN_STEPS_10 = 300_000
FULL_TRAIN_STEPS_20 = 450_000


def run_ea_job(model, grid, seed_path):
    out = os.path.join(
        MODEL_DIR,
        f"ea_g{grid}_from_{os.path.splitext(os.path.basename(seed_path))[0]}_{model}.pt"
    )

    if os.path.exists(out):
        print(f"[SKIP] EA {model} g{grid} already exists -> {out}", flush=True)
        return

    if not os.path.exists(seed_path):
        print(f"[WARN] Missing seed for EA {model} g{grid}: {seed_path}", flush=True)
        return

    print(f"[EA] Running EA {model.upper()} g{grid} from seed {seed_path}", flush=True)
    run_ea_best_observed(model, grid, seed_path, seed=1337)
    print(f"[EA] Finished EA {model.upper()} g{grid}", flush=True)


def run_parallel_batches(mlp10, cnn10, c51_10, mlp20, cnn20, c51_20, ea_parallel: bool):
    # Batch A: EA 20x20 MLP + CNN
    batch_A = [("mlp", 20, mlp20), ("cnn", 20, cnn20)]
    # Batch B: EA 10x10 MLP + CNN
    batch_B = [("mlp", 10, mlp10), ("cnn", 10, cnn10)]
    # Batch C: C51 sequential
    batch_C = [("c51", 10, c51_10), ("c51", 20, c51_20)]

    # IMPORTANT: use spawn context for CUDA safety
    ctx = mp.get_context("spawn")

    def run_batch_parallel(batch):
        procs = []
        for job in batch:
            p = ctx.Process(target=run_ea_job, args=job)
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    def run_batch_sequential(batch):
        for job in batch:
            run_ea_job(*job)

    print("\n=== EA Batch A (mlp20, cnn20) ===", flush=True)
    (run_batch_parallel if ea_parallel else run_batch_sequential)(batch_A)

    print("\n=== EA Batch B (mlp10, cnn10) ===", flush=True)
    (run_batch_parallel if ea_parallel else run_batch_sequential)(batch_B)

    print("\n=== EA Batch C (sequential: c51 10x10, 20x20) ===", flush=True)
    for job in batch_C:
        run_ea_job(*job)


def run_pipeline(
    train_batches,
    train_envs,
    train_lrs,
    c51_lrs,
    c51_envs,
    c51_atoms,
    c51_vrange,
    full_steps_10,
    full_steps_20,
    log_every,
    curiosity_every,
    ea_parallel,
):
    print("\n================ PIPELINE START ================\n", flush=True)

    # -------------------------------------------------
    # PHASE 1 — TRAIN 10×10
    # -------------------------------------------------
    print("\n================ TRAIN 10×10 =================\n", flush=True)

    best_mlp = benchmark_training_grid(
        "mlp", 10, train_batches, train_envs, train_lrs, 1,
        os.path.join(BENCH_DIR, "mlp_g10"),
        train_mlp, train_cnn, make_eval_env
    )

    best_cnn = benchmark_training_grid(
        "cnn", 10, train_batches, train_envs, train_lrs, 1,
        os.path.join(BENCH_DIR, "cnn_g10"),
        train_mlp, train_cnn, make_eval_env
    )

    mlp10 = os.path.join(MODEL_DIR, "mlp_10x10.pt")
    if not os.path.exists(mlp10):
        train_mlp(
            grid=10,
            steps=full_steps_10,
            ckpt_path=os.path.join(CKPT_DIR, "final_mlp_10x10.pt"),
            batch_size=best_mlp["batch"],
            n_envs=best_mlp["n_envs"],
            lr=best_mlp["lr"],
            do_early_stop=True,
            log_every=log_every,
            curiosity_every=curiosity_every,
        )

    cnn10 = os.path.join(MODEL_DIR, "cnn_10x10.pt")
    if not os.path.exists(cnn10):
        train_cnn(
            grid=10,
            steps=full_steps_10,
            ckpt_path=os.path.join(CKPT_DIR, "final_cnn_10x10.pt"),
            batch_size=best_cnn["batch"],
            n_envs=best_cnn["n_envs"],
            lr=best_cnn["lr"],
            do_early_stop=True,
            log_every=log_every,
            curiosity_every=curiosity_every,
        )

    c51_10 = os.path.join(MODEL_DIR, "c51_cnn_10x10.pt")
    if not os.path.exists(c51_10):
        print("\n================ BENCHMARK C51 10×10 ================\n", flush=True)
        best_c51 = None
        best_p50 = -1e9

        c51_bench_dir = os.path.join(BENCH_DIR, "c51_g10")
        os.makedirs(c51_bench_dir, exist_ok=True)

        for lr in c51_lrs:
            for E in c51_envs:
                for atoms in c51_atoms:
                    for vmin, vmax in c51_vrange:
                        ckpt = os.path.join(
                            c51_bench_dir,
                            f"p1_c51_g10_A{atoms}_V{vmin}_{vmax}_E{E}_LR{lr}.pt"
                        )

                        if os.path.exists(ckpt):
                            m = torch.load(ckpt, map_location="cpu")
                        else:
                            m = train_c51_cnn(
                                grid=10,
                                steps=C51_PHASE1_STEPS,
                                ckpt_path=ckpt,
                                n_envs=E,
                                lr=lr,
                                n_atoms=atoms,
                                v_min=vmin,
                                v_max=vmax,
                                is_benchmark=True,
                                log_every=log_every,
                                curiosity_every=curiosity_every,
                            )
                            torch.save(m, ckpt)

                        p50 = float(m["eval_p50"])
                        if p50 > best_p50:
                            best_p50 = p50
                            best_c51 = {"lr": lr, "n_envs": E, "atoms": atoms, "vmin": vmin, "vmax": vmax}

        print(f"\nBEST C51 CONFIG: {best_c51} (p50={best_p50:.2f})\n", flush=True)

        train_c51_cnn(
            grid=10,
            steps=full_steps_10,
            ckpt_path=os.path.join(CKPT_DIR, "final_c51_cnn_10x10.pt"),
            n_envs=best_c51["n_envs"],
            lr=best_c51["lr"],
            n_atoms=best_c51["atoms"],
            v_min=best_c51["vmin"],
            v_max=best_c51["vmax"],
            do_early_stop=True,
            log_every=log_every,
            curiosity_every=curiosity_every,
        )
    else:
        best_c51 = {"lr": 2e-4, "n_envs": 512, "atoms": 51, "vmin": -20, "vmax": 20}

    # -------------------------------------------------
    # PHASE 2 — TRAIN 20×20 (SEEDED)
    # -------------------------------------------------
    print("\n================ TRAIN 20×20 (SEEDED) =================\n", flush=True)

    mlp20 = os.path.join(MODEL_DIR, "mlp_20x20_seed_mlp_10x10.pt")
    if not os.path.exists(mlp20):
        train_mlp(
            grid=20,
            steps=full_steps_20,
            ckpt_path=os.path.join(CKPT_DIR, "final_mlp_20x20_seed_mlp_10x10.pt"),
            batch_size=best_mlp["batch"],
            n_envs=best_mlp["n_envs"],
            lr=best_mlp["lr"],
            seed_ckpt=mlp10,
            do_early_stop=True,
            log_every=log_every,
            curiosity_every=curiosity_every,
        )

    cnn20 = os.path.join(MODEL_DIR, "cnn_20x20_seed_cnn_10x10.pt")
    if not os.path.exists(cnn20):
        train_cnn(
            grid=20,
            steps=full_steps_20,
            ckpt_path=os.path.join(CKPT_DIR, "final_cnn_20x20_seed_cnn_10x10.pt"),
            batch_size=best_cnn["batch"],
            n_envs=best_cnn["n_envs"],
            lr=best_cnn["lr"],
            seed_ckpt=cnn10,
            do_early_stop=True,
            log_every=log_every,
            curiosity_every=curiosity_every,
        )

    c51_20 = os.path.join(MODEL_DIR, "c51_cnn_20x20_seed_c51_cnn_10x10.pt")
    if not os.path.exists(c51_20):
        train_c51_cnn(
            grid=20,
            steps=full_steps_20,
            ckpt_path=os.path.join(CKPT_DIR, "final_c51_cnn_20x20_seed_c51_cnn_10x10.pt"),
            n_envs=best_c51["n_envs"],
            lr=best_c51["lr"],
            n_atoms=best_c51["atoms"],
            v_min=best_c51["vmin"],
            v_max=best_c51["vmax"],
            seed_ckpt=c51_10,
            do_early_stop=True,
            log_every=log_every,
            curiosity_every=curiosity_every,
        )

    # -------------------------------------------------
    # PHASE 3 — EA
    # -------------------------------------------------
    print("\n================ RUNNING ALL EA JOBS (PARALLEL BATCHES) ================\n", flush=True)
    run_parallel_batches(mlp10, cnn10, c51_10, mlp20, cnn20, c51_20, ea_parallel=ea_parallel)

    print("\n================ PIPELINE COMPLETE ================\n", flush=True)


def _parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--train-envs", type=str, default=",".join(map(str, DEFAULT_TRAIN_ENVS)))
    ap.add_argument("--c51-envs", type=str, default=",".join(map(str, DEFAULT_C51_ENVS)))
    ap.add_argument("--train-batches", type=str, default=",".join(map(str, DEFAULT_TRAIN_BATCHES)))
    ap.add_argument("--train-lrs", type=str, default=",".join(map(str, DEFAULT_TRAIN_LRS)))
    ap.add_argument("--full-steps-10", type=int, default=FULL_TRAIN_STEPS_10)
    ap.add_argument("--full-steps-20", type=int, default=FULL_TRAIN_STEPS_20)

    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--curiosity-every", type=int, default=1)

    # NEW:
    ap.add_argument("--ea-parallel", action="store_true", help="Run EA batches in parallel (uses spawn). Default is sequential (recommended on 1 GPU).")

    args = ap.parse_args()

    # NOTE: do NOT force global spawn here; we only use spawn context for EA.
    run_pipeline(
        train_batches=_parse_int_list(args.train_batches),
        train_envs=_parse_int_list(args.train_envs),
        train_lrs=_parse_float_list(args.train_lrs),
        c51_lrs=DEFAULT_C51_LRS,
        c51_envs=_parse_int_list(args.c51_envs),
        c51_atoms=DEFAULT_C51_ATOMS,
        c51_vrange=DEFAULT_C51_VRANGE,
        full_steps_10=args.full_steps_10,
        full_steps_20=args.full_steps_20,
        log_every=args.log_every,
        curiosity_every=args.curiosity_every,
        ea_parallel=args.ea_parallel,
    )