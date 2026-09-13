import json
import math
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit

from config import (
    BATCH_SIZE,
    CHECKPOINT_JSON,
    DATA_DIR,
    DELTA_T_VAR,
    JSR_W,
    LOG_FILE,
    N_JOBS,
    N_SEEDS,
    OUTPUT_CSV,
    SPATIAL_VAR_CONSENSUS_THRESH,
    T,
    VAR_TEMP_CYCLE_THRESH,
    VAR_TEMP_DISSENSUS_THRESH,
    WINDOW,
)


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line)


def get_param_hash(model, topology, N, P, gamma_stub, shape, K_frac, seed):
    return f"{model}_{topology}_{N}_{P:.2f}_{gamma_stub:.2f}_{shape}_{K_frac}_{seed}"


def load_checkpoint():
    if CHECKPOINT_JSON.exists():
        try:
            with open(CHECKPOINT_JSON, "r", encoding="utf-8") as handle:
                return set(json.load(handle))
        except OSError as exc:
            log(f"Warning: could not read checkpoint ({exc}). Starting fresh.")
    return set()


def save_checkpoint_atomic(done_set):
    with tempfile.NamedTemporaryFile("w", dir=DATA_DIR, delete=False, encoding="utf-8") as handle:
        json.dump(list(done_set), handle)
        temp_name = handle.name
    os.replace(temp_name, CHECKPOINT_JSON)


def append_results_to_csv(results_list):
    valid = [row for row in results_list if row is not None]
    if not valid:
        return
    frame = pd.DataFrame(valid)
    frame.to_csv(OUTPUT_CSV, mode="a", header=not OUTPUT_CSV.exists(), index=False)


def get_file_size_mb(filepath):
    if filepath.exists():
        return filepath.stat().st_size / (1024 * 1024)
    return 0.0


def generate_graph(N, topology, seed):
    safe_seed = int(seed % (2**32))
    connected = False
    attempts = 0

    while not connected and attempts < 100:
        current_seed = safe_seed + attempts
        if topology == "ER":
            graph = nx.erdos_renyi_graph(N, 0.2, seed=current_seed)
        elif topology == "BA":
            graph = nx.barabasi_albert_graph(N, min(2, N - 1), seed=current_seed)
        elif topology == "WS":
            k = min(4, N - 1)
            if k % 2 != 0:
                k = max(2, k - 1)
            graph = nx.connected_watts_strogatz_graph(N, k, 0.1, seed=current_seed)
        elif topology == "k-regular":
            k = min(4, N - 1)
            if (N * k) % 2 != 0:
                k = max(2, k - 1)
            try:
                graph = nx.random_regular_graph(k, N, seed=current_seed)
            except nx.NetworkXError:
                graph = nx.random_regular_graph(2, N, seed=current_seed)
        else:
            raise ValueError(f"Unknown topology: {topology}")

        connected = nx.is_connected(graph)
        attempts += 1

    adj = nx.to_numpy_array(graph, dtype=np.float64)
    np.fill_diagonal(adj, 1.0)
    return adj


@njit(cache=True, fastmath=True)
def row_normalise(W):
    N = W.shape[0]
    for i in range(N):
        row_sum = 0.0
        for j in range(N):
            row_sum += W[i, j]
        if row_sum > 1e-14:
            for j in range(N):
                W[i, j] /= row_sum
        else:
            for j in range(N):
                W[i, j] = 0.0
            W[i, i] = 1.0
    return W


@njit(cache=True, fastmath=True)
def hajnal_coeff(W):
    N = W.shape[0]
    max_diff = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            diff = 0.0
            for k in range(N):
                diff += abs(W[i, k] - W[j, k])
            if diff > max_diff:
                max_diff = diff
    return 0.5 * max_diff


@njit(cache=True, fastmath=True)
def compute_penalty_factor(x, f_out, P, shape, symmetric, is_kin):
    N = len(x)
    total = 0.0
    for i in range(N):
        total += x[i]
    mean_x = total / N

    var_sum = 0.0
    for i in range(N):
        diff = x[i] - mean_x
        var_sum += diff * diff
    std_x = math.sqrt(var_sum / N)

    if std_x < 1e-7:
        for i in range(N):
            f_out[i] = 1.0
        return

    for i in range(N):
        if is_kin[i]:
            f_out[i] = 1.0
            continue

        D = (x[i] - mean_x) / (std_x + 1e-8)
        val = abs(D) if symmetric else D

        if val > 0.0:
            if shape == 0:
                h = val
            elif shape == 1:
                h = 1.0 if val > 0.5 else 0.0
            else:
                h = math.tanh(val)
            f_out[i] = max(0.0, 1.0 - P * h)
        else:
            f_out[i] = 1.0


@njit(cache=True, fastmath=True)
def proposed_step(x, x0, W0_norm, gamma_stub, is_kin, W_out, x_out, f, H_out):
    N = len(x)
    for i in range(N):
        for j in range(N):
            if W0_norm[i, j] > 0.0:
                if is_kin[i] and is_kin[j]:
                    W_out[i, j] = W0_norm[i, j]
                else:
                    W_out[i, j] = W0_norm[i, j] * f[j]
            else:
                W_out[i, j] = 0.0

    for i in range(N):
        row_sum = 0.0
        for j in range(N):
            row_sum += W_out[i, j]
        if row_sum > 1e-14:
            for j in range(N):
                W_out[i, j] /= row_sum
        else:
            for j in range(N):
                W_out[i, j] = 0.0
            W_out[i, i] = 1.0

    for i in range(N):
        dot = 0.0
        for j in range(N):
            val = W_out[i, j]
            dot += val * x[j]
            H_out[i, j] = (1.0 - gamma_stub) * val
            if i == j:
                H_out[i, j] += gamma_stub
        x_out[i] = gamma_stub * x0[i] + (1.0 - gamma_stub) * dot


@njit(cache=True, fastmath=True)
def anti_expert_step(x, x0, W0_norm, gamma_stub, history_x, t, window, W_out, x_out, H_out):
    N = len(x)
    hist_idx = t % window
    for i in range(N):
        history_x[i, hist_idx] = x[i]
    effective = t + 1 if t < window else window

    for i in range(N):
        for j in range(N):
            if W0_norm[i, j] > 0.0:
                if effective < 2:
                    W_out[i, j] = W0_norm[i, j]
                else:
                    total = 0.0
                    total_sq = 0.0
                    for k in range(effective):
                        val = history_x[j, k]
                        total += val
                        total_sq += val * val
                    mean = total / effective
                    var = max((total_sq / effective) - (mean * mean), 1e-6)
                    W_out[i, j] = W0_norm[i, j] * (1.0 / var)
            else:
                W_out[i, j] = 0.0

    for i in range(N):
        row_sum = 0.0
        for j in range(N):
            row_sum += W_out[i, j]
        if row_sum > 1e-14:
            for j in range(N):
                W_out[i, j] /= row_sum
        else:
            for j in range(N):
                W_out[i, j] = 0.0
            W_out[i, i] = 1.0

    for i in range(N):
        dot = 0.0
        for j in range(N):
            val = W_out[i, j]
            dot += val * x[j]
            H_out[i, j] = (1.0 - gamma_stub) * val
            if i == j:
                H_out[i, j] += gamma_stub
        x_out[i] = gamma_stub * x0[i] + (1.0 - gamma_stub) * dot


@njit(cache=True, fastmath=True)
def inverse_reputation_step(
    x, x0, W0_norm, true_mu, gamma_stub, history_dev, t, window, W_out, x_out, H_out
):
    N = len(x)
    hist_idx = t % window
    for i in range(N):
        history_dev[i, hist_idx] = abs(x[i] - true_mu)
    effective = t + 1 if t < window else window

    for i in range(N):
        for j in range(N):
            if W0_norm[i, j] > 0.0:
                total = 0.0
                for k in range(effective):
                    total += history_dev[j, k]
                mad = total / effective
                W_out[i, j] = W0_norm[i, j] * (1.0 / (1.0 + mad))
            else:
                W_out[i, j] = 0.0

    for i in range(N):
        row_sum = 0.0
        for j in range(N):
            row_sum += W_out[i, j]
        if row_sum > 1e-14:
            for j in range(N):
                W_out[i, j] /= row_sum
        else:
            for j in range(N):
                W_out[i, j] = 0.0
            W_out[i, i] = 1.0

    for i in range(N):
        dot = 0.0
        for j in range(N):
            val = W_out[i, j]
            dot += val * x[j]
            H_out[i, j] = (1.0 - gamma_stub) * val
            if i == j:
                H_out[i, j] += gamma_stub
        x_out[i] = gamma_stub * x0[i] + (1.0 - gamma_stub) * dot


@njit(cache=True, fastmath=True)
def stubborn_mix_step(x, x0, W0_norm, gamma_vec, x_out, H_out):
    N = len(x)
    for i in range(N):
        dot = 0.0
        for j in range(N):
            val = W0_norm[i, j]
            dot += val * x[j]
            H_out[i, j] = (1.0 - gamma_vec[i]) * val
            if i == j:
                H_out[i, j] += gamma_vec[i]
        x_out[i] = gamma_vec[i] * x0[i] + (1.0 - gamma_vec[i]) * dot


def run_single_simulation(model, topology, N, P, gamma_stub, shape, K_frac, seed, warm_up=False):
    rng = np.random.default_rng(seed)
    adj = generate_graph(N, topology, seed)
    W0_norm = adj.copy()
    row_normalise(W0_norm)

    x0 = rng.uniform(-1.0, 1.0, size=N)
    true_mu = float(np.mean(x0))

    if K_frac == "N":
        K_actual = N
    elif K_frac == "N/2":
        K_actual = max(1, N // 2)
    elif K_frac == "N/5":
        K_actual = max(1, N // 5)
    elif K_frac == "N/10":
        K_actual = max(1, N // 10)
    else:
        K_actual = 0

    is_kin = np.zeros(N, dtype=np.bool_)
    if K_actual > 0 and model in ["proposed", "symmetric", "fixed_suppression"]:
        kin_idx = rng.choice(N, size=K_actual, replace=False)
        is_kin[kin_idx] = True

    gamma_vec = np.full(N, gamma_stub, dtype=np.float64)
    if model == "stubborn-mix":
        gamma_vec = np.zeros(N, dtype=np.float64)
        stubborn_idx = rng.choice(N, size=N // 2, replace=False)
        gamma_vec[stubborn_idx] = 1.0

    history_buffer = np.zeros((N, WINDOW), dtype=np.float64)
    x_buffer = np.zeros((DELTA_T_VAR, N), dtype=np.float64)

    x_curr = x0.copy()
    x_next = np.empty_like(x_curr)
    W = np.empty((N, N), dtype=np.float64)
    H_true = np.empty((N, N), dtype=np.float64)
    f = np.ones(N, dtype=np.float64)
    fixed_weights = np.ones(N, dtype=np.float64)

    if model == "fixed_suppression":
        compute_penalty_factor(x0, fixed_weights, P, shape, False, is_kin)

    T_run = 2 if warm_up else T
    W_last100_arr = np.empty((JSR_W, N, N), dtype=np.float64)

    for t in range(T_run):
        if not warm_up and t >= (T_run - DELTA_T_VAR):
            x_buffer[t % DELTA_T_VAR, :] = x_curr

        if model == "proposed":
            compute_penalty_factor(x_curr, f, P, shape, False, is_kin)
            proposed_step(x_curr, x0, W0_norm, gamma_stub, is_kin, W, x_next, f, H_true)
        elif model == "symmetric":
            compute_penalty_factor(x_curr, f, P, shape, True, is_kin)
            proposed_step(x_curr, x0, W0_norm, gamma_stub, is_kin, W, x_next, f, H_true)
        elif model == "fixed_suppression":
            proposed_step(x_curr, x0, W0_norm, gamma_stub, is_kin, W, x_next, fixed_weights, H_true)
        elif model == "anti-expert":
            anti_expert_step(x_curr, x0, W0_norm, gamma_stub, history_buffer, t, WINDOW, W, x_next, H_true)
        elif model == "inverse-reputation":
            inverse_reputation_step(
                x_curr, x0, W0_norm, true_mu, gamma_stub, history_buffer, t, WINDOW, W, x_next, H_true
            )
        elif model == "stubborn-mix":
            stubborn_mix_step(x_curr, x0, W0_norm, gamma_vec, x_next, H_true)

        x_curr, x_next = x_next, x_curr

        if not warm_up and t >= (T_run - JSR_W):
            W_last100_arr[t % JSR_W, :, :] = H_true

    if warm_up:
        return None

    Var_temp = float(np.mean(np.var(x_buffer, axis=0)))
    terminal_spatial_var = float(np.var(x_curr))
    consensus_error = float(abs(np.mean(x_curr) - true_mu))

    M = np.eye(N)
    for i in range(JSR_W):
        idx = (T_run - JSR_W + i) % JSR_W
        M = W_last100_arr[idx] @ M
    delta_product = hajnal_coeff(M)

    if Var_temp >= VAR_TEMP_CYCLE_THRESH:
        regime = "Regime 3 (Non-convergent Cycling)"
    elif terminal_spatial_var < SPATIAL_VAR_CONSENSUS_THRESH:
        regime = "Regime 1 (Biased Convergence)"
    elif Var_temp < VAR_TEMP_DISSENSUS_THRESH and terminal_spatial_var >= SPATIAL_VAR_CONSENSUS_THRESH:
        regime = "Regime 1B (Fixed Dissensus)"
    else:
        regime = "Regime 2 (Slow Mixing)"

    return {
        "model": model,
        "topology": topology,
        "N": N,
        "P": P,
        "gamma_stub": gamma_stub,
        "penalty_shape": ["Linear", "Step", "Tanh"][shape]
        if model in ["proposed", "symmetric", "fixed_suppression"]
        else "None",
        "K_fraction": K_frac if model in ["proposed", "symmetric", "fixed_suppression"] else "None",
        "K_count": K_actual,
        "seed": seed,
        "delta_product": delta_product,
        "Var_temp": Var_temp,
        "terminal_spatial_var": terminal_spatial_var,
        "consensus_error": consensus_error,
        "regime": regime,
    }


def build_pruned_grid():
    models = [
        "proposed",
        "anti-expert",
        "inverse-reputation",
        "stubborn-mix",
        "symmetric",
        "fixed_suppression",
    ]
    topologies = ["ER", "BA", "WS", "k-regular"]
    N_vals = [20, 50, 100, 200]
    P_vals = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    gamma_vals = [0.0, 0.1, 0.2, 0.5]
    shape_vals = [0, 1, 2]
    K_fractions = ["N", "N/2", "N/5", "N/10"]

    runs = []
    for model in models:
        for topo in topologies:
            for N in N_vals:
                if model in ["proposed", "symmetric", "fixed_suppression"]:
                    for P in P_vals:
                        shapes = [0] if P == 0.0 else shape_vals
                        k_fracs = ["N"] if P == 0.0 else K_fractions
                        for K_frac in k_fracs:
                            if K_frac == "N" and P > 0.0:
                                continue
                            for gamma in gamma_vals:
                                for shape in shapes:
                                    for seed in range(N_SEEDS):
                                        runs.append((model, topo, N, P, gamma, shape, K_frac, seed))
                elif model == "stubborn-mix":
                    for seed in range(N_SEEDS):
                        runs.append((model, topo, N, 0.0, 0.0, 0, "None", seed))
                else:
                    for gamma in gamma_vals:
                        for seed in range(N_SEEDS):
                            runs.append((model, topo, N, 0.0, gamma, 0, "None", seed))
    return runs


def main():
    log("Starting simulation engine")
    log(f"Data directory: {DATA_DIR}")

    all_runs = build_pruned_grid()
    log(f"Total parameter combinations: {len(all_runs):,}")

    log("Warming up numba kernels")
    for model in ["proposed", "anti-expert", "inverse-reputation", "stubborn-mix"]:
        run_single_simulation(model, "ER", 20, 0.1, 0.1, 0, "N", 42, warm_up=True)
    log("Kernel warm-up complete")

    completed_hashes = load_checkpoint()
    remaining = [run for run in all_runs if get_param_hash(*run) not in completed_hashes]
    log(f"Runs remaining: {len(remaining):,}")

    n_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()

    for batch_idx in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_idx : batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        log(f"Executing batch {batch_num}/{n_batches} ({len(batch)} configurations)")

        batch_start = time.time()
        results = Parallel(n_jobs=N_JOBS)(
            delayed(run_single_simulation)(*params) for params in batch
        )
        batch_duration = time.time() - batch_start

        append_results_to_csv(results)
        for params in batch:
            completed_hashes.add(get_param_hash(*params))
        save_checkpoint_atomic(completed_hashes)

        file_mb = get_file_size_mb(OUTPUT_CSV)
        log(
            f"Batch {batch_num} saved in {batch_duration:.1f}s. "
            f"CSV size: {file_mb:.2f} MB. Total complete: {len(completed_hashes):,}"
        )

    total_time = (time.time() - start_time) / 3600
    log(f"Simulation complete. Elapsed time: {total_time:.2f} hours")


if __name__ == "__main__":
    main()
