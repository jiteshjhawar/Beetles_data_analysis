# Beetles Data Analysis

## Overview
This repository contains code, data, and analysis pipelines for the project.  
Some large binary files (movies, data arrays, etc.) are tracked using **Git LFS**.

---

## 🚀 Getting Started

### 1. Install Git LFS
This repository uses Git Large File Storage (LFS) for large files (>100 MB).

```bash
git lfs install
2. Clone the Repository
bash
Copy code
git clone https://github.com/jiteshjhawar/Beetles_data_analysis.git
cd Beetles_data_analysis
If you cloned earlier without LFS, fetch large files using:

bash
Copy code
git lfs pull
📦 Dependencies
Below are the Python packages required to run the analysis.

Standard Imports
python
Copy code
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
python
Copy code
from scipy.optimize import curve_fit
from numba import vectorize, float64
from statsmodels.tsa.stattools import acf
📚 Requirements File
Install all required packages using:

bash
Copy code
pip install -r requirements.txt
Typical requirements.txt:

nginx
Copy code
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

bash
Copy code
python codes/main.py
📁 Repository Structure
bash
Copy code
repo/
│── data/                 # Large data files (LFS tracked)
│── movies/               # Movie files (LFS tracked)
│── codes/                # Python analysis scripts
│   ├── functions.py
│   ├── dataFunctions.py
│   ├── Test.ipynb
│── figures/
│── requirements.txt
│── README.md
📝 Notes
Files larger than 100 MB must use Git LFS

GitHub warns for files above 50 MB (normal)

If contributing, please install Git LFS first

🤝 Contributing
Pull requests and issues are welcome!