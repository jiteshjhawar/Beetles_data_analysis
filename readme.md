Beetles Data Analysis
Overview

This repository contains code, data, and analysis pipelines for the project.
Some large binary files (movies, data arrays, etc.) are tracked using Git LFS.

🚀 Getting Started
1. Install Git LFS

This repository uses Git Large File Storage (LFS) for large files (>100 MB).

git lfs install

2. Clone the repository
git clone https://github.com/<username>/<repo>.git
cd <repo>

3. If you cloned earlier without LFS

Fetch large files:

git lfs pull

📦 Dependencies

Below are the Python packages required to run the analysis.

Standard Imports
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.backends.backend_pdf
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import functions as fn
import dataFunctions as dF
import pandas as pd
import imageio

Additional Dependencies
from scipy.optimize import curve_fit
from numba import vectorize, float64
from statsmodels.tsa.stattools import acf

📚 Requirements File

Install all required packages using:

pip install -r requirements.txt


Typical requirements.txt:

numpy
pandas
matplotlib
imageio
tqdm
scipy
numba
statsmodels

▶️ Running the Code

Example usage:

python codes/main.py &

📁 Repository Structure (example)
repo/
│── data/                 # Large data files (LFS tracked)
│── movies/               # Movie files (LFS tracked)
│── codes/                # Python analysis scripts
│     ├── functions.py
│     ├── dataFunctions.py
│     ├── Test.ipynb
│── figures/
│── requirements.txt
│── README.md

📝 Notes

Files larger than 100 MB must use Git LFS.

GitHub will reject pushes with oversized files if they aren’t handled by LFS.

If contributing, please install Git LFS first.

🤝 Contributing

Pull requests and issues are welcome!