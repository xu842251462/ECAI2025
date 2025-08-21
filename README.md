# ECAI2025

MERINDA: FPGA-Accelerated Model Recovery for Mission-Critical Autonomous Systems (ECAI 2025)

Model Recovery (MR) enables safe and explainable decision-making in mission-critical autonomous systems (MCAS) by learning governing equations of dynamics. However, traditional MR implementations rely on Neural ODEs, whose iterative solvers are inefficient on edge devices such as FPGAs. In this setting, memory footprint and energy consumption are key bottlenecks for real-time deployment.

We introduce MERINDA, a novel FPGA-accelerated MR framework that replaces iterative NODE solvers with a parallelizable neural architecture. MERINDA achieves:

11× lower DRAM usage compared to mobile GPUs

2.21× faster runtime on FPGA deployment

A demonstrated inverse relationship between memory and energy at fixed accuracy

These results highlight MERINDA’s potential for resource-constrained, real-time MCAS, bridging the gap between explainable AI and hardware-efficient deployment.
