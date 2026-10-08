#!/usr/bin/env python3
"""Falsification screen for pipeline 04; no hallucination labels are assigned."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg, sparse


def load_circuit(root):
    meta = pd.read_feather(root / "data/banc_888_meta.feather")
    kc = meta.cell_class.eq("kenyon_cell")
    dan = meta.cell_class.eq("mushroom_body_dopaminergic_neuron")
    mbon = meta.cell_type.astype(str).str.startswith("MBON")
    apl = meta.cell_type.astype(str).str.contains("APL", case=False, na=False)
    mb = meta[kc | dan | mbon | apl].copy()
    mb["nt"] = mb.neurotransmitter_predicted.fillna("acetylcholine")
    mb.loc[mb.cell_type.astype(str).str.contains("APL", na=False), "nt"] = "gaba"
    edges = pd.read_feather(root / "data/banc_888_edgelist_simple_v2.feather")
    edges = edges[edges.pre.isin(mb.root_id) & edges.post.isin(mb.root_id)]
    # Preserve the original selection, including its non-KC selection issue.
    touching = edges[edges.pre.isin(meta[kc].root_id) | edges.post.isin(meta[kc].root_id)]
    selected = set(touching.pre.value_counts().head(400).index)
    ids = sorted(set(meta[mbon | dan | apl].root_id) | selected)
    lookup = {nid: i for i, nid in enumerate(ids)}
    edges = edges[edges.pre.isin(ids) & edges.post.isin(ids)]
    pre = edges.pre.map(lookup).to_numpy(dtype=int)
    post = edges.post.map(lookup).to_numpy(dtype=int)
    counts = edges["count"].to_numpy(dtype=float)
    assert len(set(zip(pre, post))) == len(pre), "Duplicate edges need aggregation"
    # Match pipeline 04: dict(zip(...)) keeps the last annotation per root.
    nts = mb.drop_duplicates("root_id", keep="last").set_index("root_id").nt.reindex(ids).to_numpy()
    signs = np.array([1.0 if nt in ("acetylcholine", "glutamate") else
                      -1.0 if nt == "gaba" else .25 if nt == "dopamine" else .5
                      for nt in nts])
    return pre, post, counts * signs[pre], nts == "gaba", ids, len(selected - set(meta[kc].root_id))


def rewire(pre, post, n, seed, swap_multiple):
    """Directed double-edge swaps; weights stay attached to presynaptic cells."""
    rng = np.random.default_rng(seed)
    target = post.copy()
    occupied = set(zip(pre, target))
    eligible = np.flatnonzero(pre != post)
    required = swap_multiple * len(eligible)
    successes = attempts = 0
    while successes < required and attempts < required * 100:
        attempts += 1
        a, b = rng.choice(eligible, 2, replace=False)
        u, v, x, y = int(pre[a]), int(target[a]), int(pre[b]), int(target[b])
        if u == x or v == y or u == y or x == v:
            continue
        if (u, y) in occupied or (x, v) in occupied:
            continue
        occupied.remove((u, v)); occupied.remove((x, y))
        occupied.add((u, y)); occupied.add((x, v))
        target[a], target[b] = y, v
        successes += 1
    assert successes == required, "Could not complete requested swaps"
    assert len(occupied) == len(pre)
    assert np.array_equal(np.bincount(post, minlength=n), np.bincount(target, minlength=n))
    return target, {"successful_swaps": successes, "attempts": attempts,
                    "changed_edge_target_fraction": float(np.mean(target != post))}


def radius(matrix, symmetric):
    if symmetric:
        n = len(matrix)
        lo = linalg.eigvalsh(matrix, subset_by_index=[0, 0])[0]
        hi = linalg.eigvalsh(matrix, subset_by_index=[n-1, n-1])[0]
        return max(abs(lo), abs(hi)) + 1e-4
    return float(np.max(np.abs(linalg.eigvals(matrix)))) + 1e-4


def evolve(matrix, initial, gains, max_steps, tol):
    states = np.tile(initial, (1, len(gains)))
    factors = np.repeat(gains, initial.shape[1])[None, :]
    operator = sparse.csr_matrix(matrix)
    for step in range(1, max_steps + 1):
        derivative = -states + np.maximum(0, np.tanh((operator @ states) * factors))
        states = np.clip(states + .1 * derivative, 0, 1)
        if step % 50 == 0:
            residuals = np.max(np.abs(-states + np.maximum(0, np.tanh((operator @ states) * factors))), axis=0)
            if np.max(residuals) < tol:
                break
    residuals = np.max(np.abs(-states + np.maximum(0, np.tanh((operator @ states) * factors))), axis=0)
    return states, residuals, step


def plot_results(frame, paired, destination):
    paired = paired.copy()
    flags = frame.groupby(["graph", "variant", "gain"]).converged.all()
    paired["both_converged"] = [bool(flags.loc[(r.graph, r.variant, r.gain)]) for r in paired.itertuples()]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    for row, variant in enumerate(frame.variant.unique()):
        for graph in sorted(frame.graph.unique()):
            color, alpha = ("#b2182b", 1) if graph == 0 else ("#888888", .3)
            label = "Empirical" if graph == 0 else "Rewired" if graph == 1 else None
            sub = frame[(frame.variant == variant) & (frame.graph == graph) & (frame.gaba == .4)]
            axes[row, 0].plot(sub.gain, sub.mean_activity.where(sub.converged), color=color, alpha=alpha, label=label)
            sub = paired[(paired.variant == variant) & (paired.graph == graph)]
            axes[row, 1].plot(sub.gain, sub.activity_change.where(sub.both_converged), color=color, alpha=alpha)
        axes[row, 0].set_title(variant + ": GABA 40% activity")
        axes[row, 1].set_title(variant + ": change from GABA 100%")
        axes[row, 0].axhline(.05, ls="--", color="black", lw=.8)
        for ax in axes[row]:
            ax.set_ylabel("Mean activity (model units)")
            ax.set_xlabel("Gain")
        axes[row, 0].legend()
    fig.suptitle("Connectivity control (panel scales differ; nonconverged points omitted)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def run(args):
    root = args.workspace.resolve()
    out = args.output or root / "qc_reports/connectome_null_control"
    out.mkdir(parents=True, exist_ok=True)
    pre, post, weights, gaba, ids, non_kc = load_circuit(root)
    n = len(ids)
    gains = np.array([.8, 1.2, 1.6, 2.05, 2.5])
    thresholds = [.02, .04, .05, .06, .08]
    initial = np.stack([np.random.RandomState(2000+i).uniform(0, .4, n)
                        for i in range(args.initializations)], axis=1)
    rows, rewiring = [], []
    start = time.time()
    for graph in range(args.nulls + 1):
        target = post
        if graph:
            target, info = rewire(pre, post, n, 7000+graph, args.swap_multiple)
            rewiring.append({"graph": graph, **info})
        matrices = {}
        for g in (1.0, .4):
            modified = weights.copy()
            modified[gaba[pre]] *= g
            w = np.zeros((n, n))
            w[pre, target] = modified
            matrices[g] = {"symmetric": (w + w.T)/2, "directed": w.T}
        for variant in ("symmetric_renormalized", "symmetric_fixed", "directed_fixed"):
            symmetric = variant.startswith("symmetric")
            key = "symmetric" if symmetric else "directed"
            baseline_scale = radius(matrices[1.0][key], symmetric)
            for g in (1.0, .4):
                scale = radius(matrices[g][key], symmetric) if variant.endswith("renormalized") and g != 1 else baseline_scale
                final, residuals, steps = evolve(matrices[g][key]/scale, initial, gains, args.max_steps, args.tolerance)
                for k, gain in enumerate(gains):
                    sl = slice(k*args.initializations, (k+1)*args.initializations)
                    activities = np.mean(final[:, sl], axis=0)
                    rows.append({"graph": graph, "kind": "empirical" if graph == 0 else "rewired",
                                 "variant": variant, "gaba": g, "gain": float(gain),
                                 "mean_activity": float(activities.mean()),
                                 "initialization_means": activities.tolist(),
                                 "max_residual": float(residuals[sl].max()),
                                 "converged": bool(residuals[sl].max() < args.tolerance),
                                 "max_endpoint_distance": float(np.linalg.norm(final[:, sl] - final[:, sl.start:sl.start+1], axis=0).max()),
                                 "steps": steps, "normalization_scale": float(scale),
                                 "threshold_rates": {str(t): float(np.mean(activities >= t)) for t in thresholds}})
        pd.DataFrame(rows).drop(columns=["initialization_means", "threshold_rates"]).to_csv(out / "summary.csv", index=False)
        print(json.dumps({"completed_graph": graph, "total_graphs": args.nulls+1,
                          "elapsed_seconds": round(time.time()-start, 1)}), flush=True)
    metadata = {"neurons": n, "edges": len(pre), "non_KCs_in_original_top_KC_selection": non_kc,
                "null_graphs": args.nulls, "initializations": args.initializations,
                "gains": gains.tolist(), "thresholds": thresholds,
                "dt": .1, "max_steps": args.max_steps, "residual_tolerance": args.tolerance,
                "null_seed_offset": 7000, "initial_seed_offset": 2000,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "null_model": "Directed degree-preserving double-edge swaps; source weights and transmitter signs retained. Incoming strength and cell-type mixing are not preserved.",
                "rewiring": rewiring}
    (out / "results.json").write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n")
    frame = pd.DataFrame(rows)
    paired = frame.pivot(index=["graph", "kind", "variant", "gain"], columns="gaba", values="mean_activity").reset_index()
    paired["activity_change"] = paired[.4] - paired[1.0]
    paired.to_csv(out / "paired_effects.csv", index=False)
    plot_results(frame, paired, out / "null_control.png")


def refine(args):
    root = args.workspace.resolve()
    out = args.output or root / "qc_reports/connectome_null_control"
    original = json.loads((out / "results.json").read_text())
    rows = original["results"]
    settings = original["metadata"]
    if args.initializations != settings["initializations"]:
        raise ValueError("Refinement must reuse the original initialization count")
    pre, post, weights, gaba, ids, _ = load_circuit(root)
    n = len(ids)
    if settings["rewiring"] and args.swap_multiple != settings["rewiring"][0]["successful_swaps"] // np.sum(pre != post):
        raise ValueError("Refinement must reuse the original swap count")
    initial = np.stack([np.random.RandomState(2000+i).uniform(0, .4, n)
                        for i in range(args.initializations)], axis=1)
    original_failures = sum(not r["converged"] for r in rows)
    for graph in sorted({r["graph"] for r in rows if not r["converged"]}):
        target = post if graph == 0 else rewire(pre, post, n, 7000+graph, args.swap_multiple)[0]
        for variant in ("symmetric_renormalized", "symmetric_fixed", "directed_fixed"):
            failures = [r for r in rows if r["graph"] == graph and r["variant"] == variant and not r["converged"]]
            if not failures:
                continue
            baseline = np.zeros((n, n)); baseline[pre, target] = weights
            symmetric = variant.startswith("symmetric")
            baseline = (baseline + baseline.T)/2 if symmetric else baseline.T
            baseline_scale = radius(baseline, symmetric)
            for g in (1.0, .4):
                selected = [r for r in failures if r["gaba"] == g]
                if not selected:
                    continue
                modified = weights.copy(); modified[gaba[pre]] *= g
                matrix = np.zeros((n, n)); matrix[pre, target] = modified
                matrix = (matrix + matrix.T)/2 if symmetric else matrix.T
                scale = radius(matrix, symmetric) if variant.endswith("renormalized") and g != 1 else baseline_scale
                gains = np.array([r["gain"] for r in selected])
                final, residuals, steps = evolve(matrix/scale, initial, gains, args.max_steps, args.tolerance)
                for k, record in enumerate(selected):
                    sl = slice(k*args.initializations, (k+1)*args.initializations)
                    activities = np.mean(final[:, sl], axis=0)
                    record.update(mean_activity=float(activities.mean()), initialization_means=activities.tolist(),
                                  max_residual=float(residuals[sl].max()),
                                  converged=bool(residuals[sl].max() < args.tolerance),
                                  max_endpoint_distance=float(np.linalg.norm(final[:, sl] - final[:, sl.start:sl.start+1], axis=0).max()),
                                  steps=steps, refined=True,
                                  threshold_rates={str(t):float(np.mean(activities >= t)) for t in settings["thresholds"]})
        print(json.dumps({"refined_graph": graph}), flush=True)
    settings["refinement"] = {"max_steps":args.max_steps,"initial_nonconverged_conditions":original_failures,
                               "remaining_nonconverged_conditions":sum(not r["converged"] for r in rows),
                               "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "refined_results.json").write_text(json.dumps(original, indent=2)+"\n")
    frame = pd.DataFrame(rows)
    frame.drop(columns=["initialization_means", "threshold_rates"]).to_csv(out / "refined_summary.csv", index=False)
    paired = frame.pivot(index=["graph", "kind", "variant", "gain"], columns="gaba", values="mean_activity").reset_index()
    paired["activity_change"] = paired[.4] - paired[1.0]
    paired.to_csv(out / "refined_paired_effects.csv", index=False)
    plot_results(frame, paired, out / "refined_null_control.png")
    print(json.dumps(settings["refinement"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nulls", type=int, default=20)
    parser.add_argument("--initializations", type=int, default=16)
    parser.add_argument("--swap-multiple", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--refine", action="store_true", help="Rerun only previously nonconverged conditions")
    args = parser.parse_args()
    if args.nulls < 0 or args.initializations < 1 or args.swap_multiple < 1 or args.max_steps < 50 or args.tolerance <= 0:
        parser.error("Require nonnegative null count, positive seeds/swaps/tolerance and at least 50 steps")
    refine(args) if args.refine else run(args)
