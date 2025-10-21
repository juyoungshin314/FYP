# FYP — Monte Carlo & ML for Ising Models

This repository contains code and experiments for studying 2D and 3D Ising models using Monte Carlo simulation, PCA, autoencoders (AE) and variational autoencoders (VAE). The code includes simulation scripts, PCA/K-means analysis, and inference utilities for trained models.

## Repository layout (key files)

- 2D_MonteCarlo.py — Monte Carlo simulation for the 2D Ising model.
- 3D_MC.py — Metropolis Monte Carlo simulation utilities for 3D Ising model.
- PCA_reconstruction.py — Reconstruct order parameters using PCA.
- sklearn_pca.py — Helpers to load .npy spin datasets and run PCA.
- kmeans.py — K-means / MiniBatchKMeans analysis using PCA outputs.
- AE_infer_2.py, VAE_infer_1.py, 3D_VAE_infer.py — Inference scripts using trained AE / VAE models.
- model.py, datasets.py, MCdata.py, config.py — (expected helper modules for AE/VAE; if missing, add as needed)
- requirements.txt — Python dependencies.
- .gitignore — Files and directories to ignore.
