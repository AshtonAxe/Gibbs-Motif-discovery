# Motif Discovery using Gibbs Sampling (with CNN Baseline)

This project implements a **Gibbs sampler for de novo DNA motif discovery**, applied to **CTCF binding sequences**.  
The Gibbs sampling algorithm identifies overrepresented motifs by iteratively refining motif positions and re-estimating a Position Weight Matrix (PWM).  

To benchmark performance, a **simple Convolutional Neural Network (CNN)** is included as a **modern baseline** for the same dataset, enabling direct comparison between a classical generative algorithm and a discriminative deep learning model.

---

## Overview

| Component | Description |
|------------|-------------|
| **Gibbs Sampler** | Core focus of this project. Implements the classic MCMC algorithm for motif discovery using PWM re-estimation, probabilistic sampling, and convergence checking. |
| **CNN Baseline** | PyTorch-based neural network for motif classification on the same CTCF dataset. Serves as a comparison point for the Gibbs sampler’s interpretability and performance. |
| **Dataset** | CTCF DNA sequences provided in FASTA format (small example included under `data/`). |
| **Outputs** | PWM matrix, discovered motif positions, CNN training metrics, and learned filters. |

---

## Quickstart

```bash
git clone https://github.com/Ashton_Axe/gibbs-motif-discovery.git
cd gibbs-motif-discovery
conda create -n gibbs_sampler python=3.11
conda activate gibbs_sampler
pip install scikit-learn scipy pyranges biopython pyjaspar pysam pyfaidx logomaker anndata torch
python scripts/run_gibbs.py
python scripts/run_cnn.py
