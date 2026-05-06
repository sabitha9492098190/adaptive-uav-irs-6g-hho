# Adaptive UAV–IRS 6G Network Design Using Cross-Layer Harris Hawk Optimization Under Realistic Constraints

## Introduction

The rapid evolution toward sixth-generation (6G) wireless communication demands intelligent, ultra-reliable, and low-latency connectivity capable of supporting large-scale mobility, distributed learning, and highly dynamic environments. Emerging technologies such as Unmanned Aerial Vehicles (UAVs) and Intelligent Reflecting Surfaces (IRSs) offer flexible coverage extension and programmable radio environments, while Federated Learning (FL) enables privacy-preserving intelligence at the network edge.

However, existing studies typically rely on idealized channel assumptions—perfect CSI, static propagation, zero hardware impairments, and negligible mobility effects—which significantly limits their practicality in real 6G deployments involving high-speed vehicles and intermittent links. Furthermore, current optimization frameworks rarely couple physical-layer dynamics with system-layer constraints and learning-layer behaviors, leading to inefficient resource use and degraded FL performance under realistic conditions.


## Project Structure

```
.
├── README.md                          # This file
├── Dataset_info.txt                   # Dataset documentation
├── run_experiment.py                  # Main experiment runner
├── Graph.py                           # Visualization module
├── Dataset/
│   └── Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data_20240120.csv
├── cl_hho_cafl/                       # Core module
│   ├── __init__.py
│   ├── channel.py                     # Channel modeling (Doppler, aging, impairments)
│   ├── config.py                      # Configuration parameters
│   ├── data.py                        # Data processing and FL task creation
│   ├── experiment.py                  # Experiment orchestration
│   ├── fl.py                          # Federated learning logic
│   ├── hho.py                         # Harris Hawk Optimization
│   ├── model.py                       # Neural network models
│   └── utils.py                       # Utility functions
└── artifacts/                         # Experiment outputs
    ├── metrics/                       # Performance metrics
    └── cache/                         # Cached data

```

---

## Installation

### Requirements

- Python 3.8+
- NumPy
- Matplotlib
- TensorFlow/PyTorch (for FL training)
- Pandas

### Setup

1. **Clone/Extract the Repository**:
   ```bash
   cd d:\ancy1\J_prof_Sabeetha_Manoj_ver_1_10-12-2025
   ```

2. **Install Dependencies**:
   ```bash
   pip install numpy matplotlib tensorflow pandas
   ```

3. **Verify Dataset**:
   Ensure the NGSIM dataset is present in the `Dataset/` folder.

---

## Usage

### Running Experiments

Execute the main experiment runner:

```bash
python run_experiment.py
```

This will:
1. Load and preprocess NGSIM vehicle trajectory data
2. Generate FL clients from vehicle trajectories
3. Run CL-HHO optimization for UAV placement, IRS configuration, and FL scheduling
4. Execute federated learning with realistic channel conditions
5. Generate performance metrics and visualizations

