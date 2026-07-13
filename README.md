# Cluster-Aware Matching via Laplacian Optimal Transport

This repository contains the official implementation accompanying the paper **"Cluster-Aware Matching via Laplacian Optimal Transport"**.

The repository provides fully reproducible code for all experiments, figures, and quantitative results presented in the paper. Each experiment is organized as a standalone Google Colab notebook that can be executed independently.

## Repository Structure

### 1. Resutls of the alignment experiment on the CAPOD dataset
**Notebook:** `3DShapeAlignment.ipynb`

Reproduces the alignment experiment decribed in Section 5.1 in the paper.

---

### 2. Results of RSC on the CAPOD dataset
**Notebook:** `3DShapeRSC.ipynb`

Reproduces the results of RSC applied on 3D point-clouds taken from the CAPOD dataset part of which are figures 1, 2 and 3 presented in the paper.

---

### 3.Resuts of RSC on high dimensional Stock Market data
**Notebook:** `StockMarketRSC.ipynb`

Reproduces the experiment of applying RSC on high dimensional financial data decribed in section 5.2 in the paper.
## Reproducibility

Each notebook is self-contained and includes all necessary steps to:

- load the required data,
- construct the corresponding similarity graphs,
- solve the proposed Laplacian Optimal Transport problem,
- reproduce the figures and quantitative results reported in the paper.

The notebooks are intended to be executed independently and require no additional configuration beyond the listed dependencies.

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{clusterawarematching,
  title={Cluster-Aware Matching via Laplacian Optimal Transport},
  author={...},
  journal={...},
  year={2026}
}
```

```bash
git clone https://github.com/GabrielSamberg/Matching-Clustering.git
cd Matching-Clustering
pip install -r requirements.txt
