# Welcome to the Matching + Clustering repository!
Below I will provide a short explanation of the different components of this repository for an easier use.

## The CAPOD dataset
Can be found under the Data folder.


---

## Creating the point clouds
The relevant functions can be found inside the file create_clouds.py.
Run the module for an example use and plot.


---
## Calculating L_X, L_Y and C
All the relevant functions are inside the file graph_laplacians_and_c.py.
Run the module for an example with visualization of the different C matrices (for the uniform and non uniform variant).

---

## Optimization 
Run the module optimization.py for an example run of the following procedure:
- Set the point clouds X and Y.
- Calculate L_X,L_Y and C.
- Optimize our objective and plot the optimal coupling for the uniform and non-uniform case of the marginal distributions of the transportation polytope

---
## Refined Synchronized Clustering
Run the module refined_simultaneous_clustering.py for an example run with an interactive 3D plot of the result.

---
## Stock market experiment
Here I add a link to a Google Colab notebook with a detailed walkthrough and implementation of the stock market experiment 
### 🛑 WARNING!
### At the moment the link to the colab notebook below wont work since the repo is private.
### You still have an access to the notebook .ipynb file on the repo itself.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabrielSamberg/Matching-Clustering/blob/main/M_C_finance.ipynb)


## 🚀 Installation

```bash
git clone https://github.com/GabrielSamberg/Matching-Clustering.git
cd Matching-Clustering
pip install -r requirements.txt
