---
title: Running InvokeAI on Google Colab using a Jupyter Notebook
---

## Introduction

We have a [Jupyter
notebook](https://github.com/invoke-ai/InvokeAI/blob/main/notebooks/Stable_Diffusion_AI_Notebook.ipynb)
with cell-by-cell installation steps. It will download the code in
this repo as one of the steps, so instead of cloning this repo, simply
download the notebook from the link above and load it up in VSCode
(with the appropriate extensions installed)/Jupyter/JupyterLab and
start running the cells one-by-one.

!!! Note "you will need NVIDIA drivers, Python 3.10, and Git installed beforehand"

## Running Online On Google Colabotary
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/invoke-ai/InvokeAI/blob/main/notebooks/Stable_Diffusion_AI_Notebook.ipynb)

## Running Locally (Cloning)

1. Install the Jupyter Notebook python library (one-time):
pip install jupyter

2. Clone the InvokeAI repository:
git clone https://github.com/invoke-ai/InvokeAI.git
cd invoke-ai
3. Create a virtual environment using conda:
conda create -n invoke jupyter
4. Activate the environment and start the Jupyter notebook:
conda activate invoke
jupyter notebook
