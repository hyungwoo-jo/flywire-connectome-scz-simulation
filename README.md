# FlyWire Micro-Circuit E/I Perturbation & Schizophrenia Modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

An open-source computational neuroscience repository exploring how **synapse-level Excitation/Inhibition (E/I) imbalance** (a cornerstone hypothesis in Schizophrenia pathophysiology) disrupts circuit dynamics and network topology, using the whole-brain connectome of *Drosophila melanogaster* (FlyWire / BANC).

---

## 🔬 Scientific Motivation & Research Questions

1. **Micro-to-Macro Bridge**:
   Human MRI studies (fMRI/DTI/MIND) report macroscopic connectome breakdowns, rich-club hub disintegration, and dysconnectivity in patients with Schizophrenia (SCZ). However, non-invasive imaging cannot isolate single-synapse biophysical mechanisms.
2. **In-Silico E/I Perturbation**:
   In Schizophrenia, cortical parvalbumin-positive GABAergic interneuron hypofunction leads to runaway excitation and decreased Signal-to-Noise Ratio (SNR).
3. **The FlyWire Connectome Advantage**:
   With 139,000+ identified neurons and millions of mapped synapses categorized by neurotransmitter identities (Acetylcholine, Glutamate, GABA, Dopamine, Serotonin), we can simulate in-silico lesions and targeted neuromodulatory disruptions on real biological wiring diagrams.

---

## 📂 Repository Structure

```
flywire_connectome/
├── data/                  # Downloaded Feather / TSV connectomes (.gitignore)
├── pipelines/
│   └── 01_ei_perturbation_simulation.py  # Core dynamic Wilson-Cowan rate simulation
├── qc_reports/            # Generated publication figures
│   └── flywire_scz_ei_disruption_simulation.png
├── docs/                  # Modeling notes & circuit topology references
└── README.md
```

---

## 🚀 Quickstart & Reproduction

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/flywire_connectome.git
cd flywire_connectome

# 2. Setup Virtual Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Download Open Connectome Data (Automated via GCS Public Bucket)
curl -L https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_meta.feather -o data/banc_888_meta.feather
curl -L https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_edgelist_simple_v2.feather -o data/banc_888_edgelist_simple_v2.feather

# 4. Run E/I Balance Perturbation Simulation
python pipelines/01_ei_perturbation_simulation.py
```

---

## 📊 Key Findings from Initial Simulation

- **Baseline vs. SCZ Model (GABA -40%)**:
  - In normal conditions, recurrent inhibitory loops effectively quench sensory inputs after the stimulation window, maintaining high SNR.
  - In the 40% GABA hypofunction model, background noise floods downstream circuits, leading to persistent baseline hyper-reactivity (biological noise amplification).
- **Hub Analysis**:
  - Identification of top in-degree hub neurons driven predominantly by cholinergic projections, revealing sensitive targets for network-level vulnerability.

---

## 🤝 Community Feedback & Contributions
Pull requests, issues, and peer feedback on biophysical models, spiking neural network (SNN) extensions, or network control theory formulations are warmly welcomed!
