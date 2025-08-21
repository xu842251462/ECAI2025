# ECAI2025

# MERINDA: FPGA-Accelerated Model Recovery for Mission-Critical Autonomous Systems (ECAI 2025)

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
- [PyTorch](https://pytorch.org/) >= 2.0  
- [Xilinx Vitis HLS / Vitis Unified Software Platform](https://www.xilinx.com/products/design-tools/vitis.html)  
- FPGA board (tested on **Xilinx Alveo U250** / **Zynq Z2**)  

### Setup
```bash
git clone https://github.com/xu842251462/ECAI2025.git
cd ECAI2025
pip install -r requirements.txt

## 🚀 Usage

### 1. Train the MR model
```bash
python train.py --config configs/mr_config.yaml


