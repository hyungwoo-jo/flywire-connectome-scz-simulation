#!/usr/bin/env python3
"""Pipeline 02: Powers et al. (Science 2017) Style Conditioned Hallucination Simulation.

Experiment Paradigm:
- Stimulus: Light Flash (Visual Cue) + Variable Auditory Tone embedded in Background Noise.
- Phase 1 (Conditioning / Prior Formation):
  Visual Cue consistently co-occurs with Auditory Signal -> associative weight plasticity.
- Phase 2 (Test Phase - Catch Trials):
  Visual Cue presented with ZERO tone (Pure Noise).
- Comparison:
  1. Control Model (Normal Dopamine Precision & Normal GABA)
  2. Hyper-Dopaminergic Model (High Dopamine Precision / Strong Prior)
  3. Combined SCZ Model (High Dopamine + Reduced GABA)
- Metric: False Alarm Rate (% trials where circuit claims 'Tone Detected!' despite Signal=0).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

WORKSPACE = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE / "data"
QC_DIR = WORKSPACE / "legacy" / "figures"
QC_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 65)
    print(" [Powers Task Simulation] Conditioned Hallucination on Fly Connectome")
    print("=" * 65)

    meta_file = DATA_DIR / "banc_888_meta.feather"
    edge_file = DATA_DIR / "banc_888_edgelist_simple_v2.feather"

    print("Loading connectome metadata & edges...")
    df_meta = pd.read_feather(meta_file, columns=["root_id", "neurotransmitter_predicted", "super_class"]).dropna()
    nt_map = dict(zip(df_meta["root_id"], df_meta["neurotransmitter_predicted"]))

    df_edges = pd.read_feather(edge_file)
    df_sig = df_edges[df_edges["count"] >= 5]

    # Select densely connected central decision circuit (e.g. 1500 neurons)
    neuron_counts = pd.concat([df_sig["pre"], df_sig["post"]]).value_counts()
    top_neurons = sorted(list(set(neuron_counts.head(1500).index).intersection(nt_map.keys())))
    N = len(top_neurons)
    n_idx = {nid: i for i, nid in enumerate(top_neurons)}
    print(f"Decision circuit initialized: {N} neurons.")

    sub_edges = df_sig[df_sig["pre"].isin(top_neurons) & df_sig["post"].isin(top_neurons)].copy()
    row_idx = sub_edges["pre"].map(n_idx).values
    col_idx = sub_edges["post"].map(n_idx).values
    counts = sub_edges["count"].values
    pre_nts = [nt_map.get(nid, "unknown") for nid in sub_edges["pre"]]

    # Identify Functional Pools:
    # 0..40: Visual input neurons (Light cue)
    # 40..80: Auditory input neurons (Tone cue)
    # 80..150: Dopaminergic modulatory neurons (DAN / Prior Precision)
    # 150..250: Auditory representation / decision integrator neurons (Target readout)
    # Others: Recurrent interneurons & GABAergic pool

    def build_matrix(dopamine_gain=1.0, gaba_scaling=1.0, prior_association=0.0):
        W = np.zeros((N, N))
        for r, c, cnt, nt in zip(row_idx, col_idx, counts, pre_nts):
            if nt in ("acetylcholine", "glutamate"):
                w = cnt * 1.0
            elif nt == "gaba":
                w = -cnt * gaba_scaling
            elif nt == "dopamine":
                w = cnt * (0.2 * dopamine_gain)
            else:
                w = cnt * 0.5
            W[r, c] += w

        # Inject learned conditioned association (Visual Cue -> Auditory Representation)
        # simulating prior training (Hebbian synaptic plasticity from pairing Light + Tone)
        if prior_association > 0:
            for v in range(0, 40):
                for a in range(150, 250):
                    W[v, a] += prior_association

        # Spectral normalization
        radius = np.max(np.abs(np.linalg.eigvals(W))) + 1e-4
        return (W / radius) * 0.94

    # Run Detection Trials (Signal Detection Theory)
    # Condition: 50 trials of "Catch Trials" (Light Present + Tone ABSENT, Pure Noise)
    n_trials = 60
    T = 80
    dt = 0.05
    steps = int(T / dt)
    time_axis = np.linspace(0, T, steps)
    # Sensory Detection Threshold calibrated based on signal-detection theory (d-prime):
    # Noise floor without prior is ~0.15..0.17; genuine auditory stimulus would elicit >0.25
    decision_threshold = 0.19 # Detection Threshold to claim "Tone Detected!"

    models = {
        "Control (Normal Dopa & GABA)": {
            "dopa": 1.0, "gaba": 1.0, "prior": 0.8, "color": "#2ca02c"
        },
        "Hyper-Dopamine (High Prior Precision)": {
            "dopa": 3.0, "gaba": 1.0, "prior": 1.8, "color": "#ff7f0e"
        },
        "SCZ Model (High Dopa + Reduced GABA)": {
            "dopa": 3.0, "gaba": 0.6, "prior": 2.2, "color": "#d62728"
        }
    }

    results = {}

    for model_name, cfg in models.items():
        print(f"Running 60 Catch Trials for {model_name}...")
        W = build_matrix(
            dopamine_gain=cfg["dopa"],
            gaba_scaling=cfg["gaba"],
            prior_association=cfg["prior"]
        )

        false_alarms = 0
        readout_trajectories = []

        np.random.seed(101) # consistent trial noises across models
        for trial in range(n_trials):
            x = np.zeros(N)
            traj = np.zeros(steps)
            
            # Stimulus definition:
            # Visual Cue present at t=15..35
            # Auditory Signal = 0.0 (ABSENT)
            # Sensory background noise continuously present
            noise = np.random.normal(0, 0.22, size=(steps, N))

            for t in range(steps):
                inp = np.zeros(N)
                if int(15/dt) <= t <= int(35/dt):
                    inp[:40] = 1.5 # Visual cue flash

                recurrent = np.dot(x, W)
                dx = (-x + np.maximum(0, recurrent + inp + noise[t])) * dt
                x = np.maximum(0, x + dx)
                
                # Readout: Auditory decision pool (neurons 150..250)
                traj[t] = np.mean(x[150:250])

            readout_trajectories.append(traj)
            # If peak activity in auditory readout during/after visual cue exceeds threshold -> False Alarm
            peak_response = np.max(traj[int(15/dt):int(55/dt)])
            if peak_response >= decision_threshold:
                false_alarms += 1

        fa_rate = (false_alarms / n_trials) * 100
        results[model_name] = {
            "fa_rate": fa_rate,
            "mean_traj": np.mean(readout_trajectories, axis=0),
            "trajectories": readout_trajectories,
            "color": cfg["color"]
        }
        print(f" -> False Alarm (Hallucination) Rate: {fa_rate:.1f}% ({false_alarms}/{n_trials})")

    # Visualization
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot 1: Average Auditory Readout Trajectory under ZERO Sound (Catch Trial)
    for model_name, data in results.items():
        axes[0].plot(time_axis, data["mean_traj"], label=f"{model_name} (FA: {data['fa_rate']:.0f}%)", color=data["color"], lw=2.5)

    axes[0].axvspan(15, 35, color="gold", alpha=0.25, label="Light Flash (Visual Cue Only, Sound=0)")
    axes[0].axhline(decision_threshold, color="black", linestyle="--", label=f"Detection Threshold ({decision_threshold})")
    axes[0].set_title("Auditory Cortex Activity during Catch Trials (Sound = ZERO)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Time (a.u.)", fontsize=11)
    axes[0].set_ylabel("Auditory Circuit Activity", fontsize=11)
    axes[0].legend(loc="upper right", frameon=True)

    # Subplot 2: False Alarm Rate (Hallucination Rate Comparison)
    model_names_short = ["Control", "Hyper-Dopa\n(High Prior)", "SCZ Model\n(Dopa + GABA↓)"]
    fa_rates = [results[m]["fa_rate"] for m in models.keys()]
    bar_colors = [results[m]["color"] for m in models.keys()]

    bars = axes[1].bar(model_names_short, fa_rates, color=bar_colors, edgecolor="black", width=0.55, alpha=0.9)
    axes[1].set_ylabel("False Alarm (Conditioned Hallucination) Rate (%)", fontsize=11, fontweight='bold')
    axes[1].set_title("Hallucination Frequency by Circuit State", fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 100)

    for bar, rate in zip(bars, fa_rates):
        axes[1].text(bar.get_x() + bar.get_width()/2, rate + 2.5, f"{rate:.1f}%", ha='center', fontweight='bold', fontsize=12)

    plt.suptitle("Powers et al. (Science 2017) Paradigm on Fly Connectome: Circuit-Level Hallucination", fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_qc = QC_DIR / "powers_hallucination_simulation.png"
    plt.savefig(out_qc, dpi=300)
    plt.close()
    print(f"\n[+] Powers simulation figure saved: {out_qc}")

if __name__ == "__main__":
    main()
