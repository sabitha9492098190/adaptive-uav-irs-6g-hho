# Adaptive UAV–IRS 6G Network Design Using Cross-Layer Harris Hawk Optimization Under Realistic Constraints

## Introduction

The rapid evolution toward sixth-generation (6G) wireless communication demands intelligent, ultra-reliable, and low-latency connectivity capable of supporting large-scale mobility, distributed learning, and highly dynamic environments. Emerging technologies such as Unmanned Aerial Vehicles (UAVs) and Intelligent Reflecting Surfaces (IRSs) offer flexible coverage extension and programmable radio environments, while Federated Learning (FL) enables privacy-preserving intelligence at the network edge.

However, existing studies typically rely on idealized channel assumptions—perfect CSI, static propagation, zero hardware impairments, and negligible mobility effects—which significantly limits their practicality in real 6G deployments involving high-speed vehicles and intermittent links. Furthermore, current optimization frameworks rarely couple physical-layer dynamics with system-layer constraints and learning-layer behaviors, leading to inefficient resource use and degraded FL performance under realistic conditions.


## Dataset

The code uses the **Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data** dataset, specifically the CSV file `Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data_20240120.csv` located in the `Dataset/` folder. This dataset provides real-world vehicle trajectory data collected from highways (e.g., US-101, I-80, Lankershim, Peachtree).

### Dataset Usage

- **Mobility Source**: Vehicle positions, speeds, accelerations, lane IDs, and headway information are extracted to simulate realistic vehicular mobility in the FL network.
- **FL Task Creation**: Each vehicle's trajectory is segmented into sequential windows of vehicle states (e.g., 20 frames of data). The task is **binary lane-change prediction**: predicting whether the vehicle will change lanes within the next prediction horizon (e.g., 20 frames).
- **Client Generation**: Vehicles with sufficient trajectory length are treated as FL clients, each with local datasets derived from their own driving patterns.
- **Road Geometry**: The dataset includes location-specific road layouts (e.g., US-101 with multiple lanes) used for UAV positioning and IRS placement.

The code couples this mobility data with:

- dynamic UAV placement
- IRS-assisted link modeling with quantized phase shifts and reconfiguration delay
- mobility-induced Doppler and channel aging
- communication-aware client selection
- Harris Hawk Optimization for cross-layer control

## What the code learns

The federated task is binary lane-change prediction:

- input: a sequence of recent NGSIM vehicle states (13 features per frame, including position, speed, lane, headway, etc.)
- target: whether the vehicle changes lane within the prediction horizon

Each vehicle is treated as an FL client with local sequential data generated from its own trajectory.


## Project Structure

```
.
├── README.md                          # This file
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
   cd https://github.com/sabitha9492098190/adaptive-uav-irs-6g-hho.git
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

