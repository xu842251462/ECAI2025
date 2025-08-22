# ECAI2025

# MERINDA: FPGA-Accelerated Model Recovery for Mission-Critical Autonomous Systems 

## Impact Lab, School of Computing and Augmented Intelligence, Arizona State University, Tempe, AZ, USA (https://github.com/ImpactLabASU)

Model Recovery (MR) enables **safe and explainable decision-making** in mission-critical autonomous systems (MCAS) by learning governing equations of dynamics. However, traditional MR implementations rely on Neural ODEs, whose **iterative solvers are inefficient on edge devices** such as FPGAs. In this setting, **memory footprint and energy consumption** are key bottlenecks for real-time deployment.  

We introduce **MERINDA**, a novel FPGA-accelerated MR framework that **replaces iterative NODE solvers with a parallelizable neural architecture**.  

✨ **Key Highlights:**  
- 🚀 **11× lower DRAM usage** compared to mobile GPUs  
- ⚡ **2.21× faster runtime** on FPGA deployment  
- 🔄 Reveals an **inverse relationship between memory and energy** at fixed accuracy  
- ✅ Suitable for **resource-constrained, real-time MCAS**  

---

## 📑 Paper
This repository accompanies our **ECAI 2025 paper**:  

> *MERINDA: FPGA-Accelerated Model Recovery for Mission-Critical Autonomous Systems*  
> Accepted at **ECAI 2025** (European Conference on Artificial Intelligence).  

📄 [Preprint on arXiv — to be added]  
🔗 [Publisher Link — to be added]  

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+  
- Tensorflow = 2.21  
- [Xilinx Vitis HLS / Vitis Unified Software Platform](https://www.xilinx.com/products/design-tools/vitis.html)  
- FPGA board (tested on **PYNQ-Z2**)

### Setup
```bash
git clone https://github.com/xu842251462/ECAI2025.git
cd ECAI2025
```

# GRU + SINDy FPGA Accelerator and GPU Time-Series Models

This repository provides two main components:

1. **FPGA Accelerators**: GRU-based and SINDy-based accelerators, deployable on PYNQ boards or rebuildable via Xilinx Vivado/Vitis.  
2. **GPU-Ready Models**: Python scripts for training/evaluating time-series models (LSTM, CTRNN, LTC, etc.) across multiple real-world datasets.
---

## 📂 Repository Layout

| File / Folder | Description |
|---------------|-------------|
| `acceleration_kernel.bit`, `acceleration_kernel.hwh`, `acceleration_kernel.tcl` | Bitstream, hardware handoff, and TCL script for the **GRU accelerator** overlay. |
| `sindy128.bit`, `sindy128.hwh`, `sindy128.tcl` | Bitstream, hardware handoff, and TCL script for the **SINDy** overlay variant. |
| `customized_GRU_accelerator.cpp` | C++ source to exercise the GRU accelerator (or CPU reference). |
| `main_tb.cpp` | C++ testbench / driver for CPU-only runs. |
| `main.h` | Common header file used in C++ sources. |
| `sindy.cpp` | Standalone SINDy CPU demo. |
| `kernel.py` | Python driver for PYNQ boards to load bitstreams and call the accelerator. |
| `gru_training.ipynb` | Jupyter notebook for training/evaluating the GRU model (CPU/GPU, no FPGA required). |

---

### GPU / Python Model Files

| File / Folder | Description |
|---------------|-------------|
| `ctrnn_model.py` | Continuous-Time RNN (CTRNN) training/evaluation script. |
| `ltc_model.py` | Liquid Time-Constant (LTC) model training/evaluation script. |
| `gesture.py` | Gesture recognition dataset pipeline (GPU-enabled). |
| `har.py` | Human Activity Recognition training/evaluation script. |
| `occupancy.py` | Room occupancy prediction experiment. |
| `ozone.py` | Ozone concentration forecasting. |
| `person.py` | Person/people-count prediction task. |
| `power.py` | Power consumption/load forecasting. |
| `traffic.py` | Traffic flow/volume forecasting. |
| `download_datasets.sh` | Helper script to fetch/preprocess datasets (creates `data/` folder). |

---
#Example Runs for FPGA
```bash
vivado -mode batch -source acceleration_kernel.tcl
vivado -mode batch -source sindy128.tcl
```

#Download Datasets
```bash
source download_datasets.sh
```

#Example Runs for GPU
```bash
python3 traffic.py --model ltc --epochs 300 --size 64 --log 5
```
## Results
<img width="833" height="255" alt="image" src="https://github.com/user-attachments/assets/2b42882d-59b4-4dcc-aafc-6b9d9a20abc9" />



