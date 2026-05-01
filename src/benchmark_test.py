
import gc
import io
import time
import cProfile
import pstats
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator, FixedLocator, FuncFormatter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours
# ─────────────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def c(text, colour): return f"{colour}{text}{RESET}"
def header(text): return print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}\n{BOLD}{CYAN}  {text}{RESET}\n{BOLD}{CYAN}{'─'*60}{RESET}")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────
# Wine quality dataset (Same as demo)
def _load_wine():
    df = pd.read_csv('../data/winequality-red.csv')
    df = df.drop('quality', axis = 1)
    return df

# Make synthetic cluster structure
def _make_synthetic(n_obs = 2000, n_vars = 40, n_groups = 4, seed = 42):
    rng = np.random.default_rng(seed)
    n_per_group = n_vars // n_groups
    cols = {}
    for g in range(n_groups):
        factor = rng.standard_normal(n_obs)
        for v in range(n_per_group):
            loading = rng.uniform(0.6, 0.9)
            noise = rng.standard_normal(n_obs)
            cols[f"g{g}_v{v}"] = loading * factor + np.sqrt(1 - loading ** 2) * noise
    return pd.DataFrame(cols)

# Load data for testing
def load_data(n_obs = None, n_vars = None, n_groups = None, make = True):
    if make is False:
        df = _load_wine()
    else:
        # Make big data
        df = _make_synthetic(n_obs, n_vars, n_groups)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Correctness check
# ─────────────────────────────────────────────────────────────────────────────
def check_correctness(orig_cls, opt_cls, df, tol = 1e-6):
    header("CORRECTNESS CHECK")

    vc_orig = orig_cls(df).varclus()
    vc_opt = opt_cls(df).varclus()

    info_orig = vc_orig.info.sort_values("Cluster").reset_index(drop = True)
    info_opt = vc_opt.info.sort_values("Cluster").reset_index(drop = True)

    rs_orig = vc_orig.rsquare.sort_values(["Cluster", "Variable"]).reset_index(drop = True)
    rs_opt = vc_opt.rsquare.sort_values(["Cluster", "Variable"]).reset_index(drop = True)

    checks = {
        "N clusters match": len(vc_orig.clusters) == len(vc_opt.clusters),
        "N_Vars match": (info_orig["N_Vars"].values == info_opt["N_Vars"].values).all(),
        "Eigval1 close": np.allclose(
            info_orig["Eigval1"].astype(float),
            info_opt["Eigval1"].astype(float),
            atol = tol
        ),
        "Eigval2 close": np.allclose(
            info_orig["Eigval2"].astype(float),
            info_opt["Eigval2"].astype(float),
            atol = tol
        ),
        "RS_Own close": np.allclose(
            rs_orig["RS_Own"].astype(float),
            rs_opt["RS_Own"].astype(float), 
            atol = tol
        ),
        "RS_NC close": np.allclose(
            rs_orig["RS_NC"].astype(float),
            rs_opt["RS_NC"].astype(float),
            atol = tol
        ),
        "RS_Ratio close": np.allclose(
            rs_orig["RS_Ratio"].astype(float),
            rs_opt["RS_Ratio"].astype(float),
            atol = tol
        ),
    }

    all_passed = True
    for test, passed in checks.items():
        status = c("✓ PASS", GREEN) if passed else c("✗ FAIL", RED)
        print(f"  {status}  {test}")
        if not passed:
            all_passed = False

    if all_passed:
        print(c("\n  All checks passed — outputs are numerically identical.", GREEN))
    else:
        print(c("\n  Some checks FAILED — inspect diffs above.", RED))

    return print(c("\nDONE", GREEN)) if all_passed else print(c("\nFAIL", RED))

# ─────────────────────────────────────────────────────────────────────────────
# cProfile report
# ─────────────────────────────────────────────────────────────────────────────
def run_cprofile(cls, df, label, top_n = 15):
    header(f"cPROFILE — {label}")
    pr = cProfile.Profile()
    pr.enable()
    cls(df).varclus()
    pr.disable()

    buf = io.StringIO()
    stats = pstats.Stats(pr, stream = buf).sort_stats("cumulative")

    print("  ncalls  tottime  percall  cumtime  function")
    for (filename, lineno, funcname), stat in list(stats.stats.items())[:top_n]:
        ncalls  = stat[0]
        tottime = stat[2]
        cumtime = stat[3]
        percall = tottime / ncalls if ncalls else 0
        print(f"  {ncalls:6}  {tottime:7.3f}  {percall:7.3f}  {cumtime:7.3f}  {funcname}")
    return

# ─────────────────────────────────────────────────────────────────────────────
# Time benchmark
# ─────────────────────────────────────────────────────────────────────────────
# Test run by disable Garbage Collector (GC)
def _run_timeit(fn, n_runs, label):
    # Run time function return (times_list, mean ,std)
    times = []
    for i in range(n_runs):
        gc.disable()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        gc.enable()
        times.append(t1 - t0)
        print(f"    run {i+1}/{n_runs}  {t1-t0:.4f}s", end = "\r")
    print(" " * 40, end = "\r")
    return times

# Benchmarking by n times
def benchmark_times(orig_cls, opt_cls, df, n_runs):
    # Compare original version and optimized version
    header(f"TIMING BENCHMARK BY RUN {n_runs} TIMES")

    results = {}

    for label, cls in [("Original", orig_cls), ("Optimized", opt_cls)]:
        print(c(f"\n▶ {label}", BOLD))
        times = _run_timeit(
            lambda c = cls: c(df).varclus(),
            n_runs, label
        )
        mean_t = np.mean(times)
        std_t  = np.std(times)
        min_t  = np.min(times)
        max_t  = np.max(times)
        results[label] = {"times": times, "mean": mean_t, "std": std_t,
                          "min": min_t, "max": max_t}
        print(f"  {'mean':<10} {mean_t:.4f}s ± {std_t:.4f}s")
        print(f"  {'min':<10} {min_t:.4f}s")
        print(f"  {'max':<10} {max_t:.4f}s")

    # Speedup ratio
    header("SUMMARY")
    orig_mean = results["Original"]["mean"]
    opt_mean  = results["Optimized"]["mean"]
    speedup   = orig_mean / opt_mean

    rows = []
    for label, r in results.items():
        rows.append({
            "Version"  : label,
            "Mean (s)" : f"{r['mean']:.4f}",
            "Std (s)"  : f"{r['std']:.4f}",
            "Min (s)"  : f"{r['min']:.4f}",
            "Max (s)"  : f"{r['max']:.4f}",
        })

    summary_df = pd.DataFrame(rows).set_index("Version")
    print(summary_df.to_string())

    colour = GREEN if speedup >= 1 else RED
    print(f"\n  {BOLD}Speedup: {c(f'{speedup:.2f}x', colour)}{RESET}")
    
    return results

# Plot benchmarking by n times
def plot_time(time_result, title):
    plt.figure(figsize = (10, 6))
    for label, value in time_result.items():
        plt.plot(
            value["times"],
            label = label
        )
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer = True))
        plt.title(title)
        plt.xlabel("Number of runs")
        plt.ylabel("Times(s)")
        plt.legend(frameon = True, facecolor = 'white', loc = "upper right")

    return plt.show()

# Benchmarking by scaling
def benchmark_scaling(opt_cls, dataset_sizes, n_runs = 5, n_vars= 40, n_groups = 4, seed = 42):
    header(f"SCALING BENCHMARK ({n_runs} runs per size)")

    summary_rows = []
    raw_rows = []

    for n_obs in dataset_sizes:
        for n_var in n_vars:
            print(c(f"\n▶ Data size: {n_obs}, Features: {n_var}", BOLD))

            df = _make_synthetic(
                n_obs = n_obs,
                n_vars = n_var,
                n_groups = n_groups,
                seed = seed,
            )

            times = _run_timeit(
                lambda: opt_cls(df).varclus(),
                n_runs,
                label=f"Data size: {n_obs}, Vars: {n_var}",
            )

            # For plotting
            for i, t in enumerate(times):
                raw_rows.append({
                    "n_obs": n_obs,
                    "n_var": n_var,
                    "run": i,
                    "time": t,
                })

            # Summary stats
            row = {
                "n_obs": n_obs,
                "n_var": n_var,
                "mean": np.mean(times),
                "std":  np.std(times),
                "min":  np.min(times),
                "max":  np.max(times),
            }
            summary_rows.append(row)

            print(f"  {'mean':<10} {row['mean']:.4f}s ± {row['std']:.4f}s")
            print(f"  {'min':<10} {row['min']:.4f}s")
            print(f"  {'max':<10} {row['max']:.4f}s")

    summary_df = pd.DataFrame(summary_rows)
    raw_df = pd.DataFrame(raw_rows)

    header("SUMMARY")
    print(summary_df.to_string(float_format = "%.4f", index = False))

    return raw_df

# Plot benchmarking by accumulating dataset 
def plot_scaling(time_result, title):
    fig, ax = plt.subplots(figsize = (10, 6))

    for n_var, g in time_result.groupby("n_var"):
        ax.scatter(
            g["n_obs"],
            g["time"],
            alpha = 0.6,
            label = f"n_var = {n_var}",
        )

    x_vals = sorted(time_result["n_obs"].unique())
    ax.xaxis.set_major_locator(FixedLocator(x_vals))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.set_xticks(x_vals)
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("Times(s)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    return plt.show()