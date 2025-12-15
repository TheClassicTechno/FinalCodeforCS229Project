# Environment Setup Guide

Complete setup instructions for running the CS229 Options Mispricing Detection project.

---

## Quick Summary

**What you need:**
- Python 3.9 or higher
- About 10 minutes to set up
- An internet connection (to fetch options data from Yahoo Finance)

**What you'll install:**
- NumPy, Pandas (data handling)
- scikit-learn (machine learning)
- Matplotlib, Seaborn (visualization)
- yfinance, py_vollib (options data and pricing)

---

## Setup Method 1: Using Conda (Recommended)

Conda is the easiest way to set up the environment because it handles all package dependencies automatically.

### Step 1: Install Conda

If you don't have Conda installed:
- Download Miniconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- Or use Anaconda if you already have it

### Step 2: Create the Environment

```bash
# Navigate to the project directory
cd CS229_QuantML_Project

# Create environment from file (takes 2-3 minutes)
conda env create -f environment.yml

# Activate the environment
conda activate cs229-quantml
```

That's it! You're ready to run the code.

### Step 3: Verify Installation

```bash
# Check Python version (should be 3.9+)
python --version

# Check scikit-learn (should be 1.5.0+)
python -c "import sklearn; print(sklearn.__version__)"

# Check if all packages loaded
python -c "import numpy, pandas, sklearn, matplotlib, yfinance; print('All packages loaded successfully!')"
```

---

## Setup Method 2: Using pip (Alternative)

If you prefer pip or don't want to use Conda, here's how to set up with Python's built-in virtual environment.

### Step 1: Create Virtual Environment

```bash
# Navigate to project directory
cd CS229_QuantML_Project

# Create virtual environment
python -m venv cs229_env

# Activate environment
# On macOS/Linux:
source cs229_env/bin/activate

# On Windows:
cs229_env\Scripts\activate
```

### Step 2: Install Packages

```bash
# Upgrade pip first
pip install --upgrade pip

# Install required packages
pip install numpy pandas scikit-learn scipy matplotlib seaborn yfinance py_vollib python-dateutil
```

**Package versions we tested:**
- numpy >= 1.24.0
- pandas >= 2.0.0  
- scikit-learn >= 1.5.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- yfinance >= 0.2.0
- py_vollib >= 1.0.0

### Step 3: Verify Installation

Same as Conda method above - run the verification commands.

---

## Troubleshooting Common Issues

### Issue: "ModuleNotFoundError: No module named 'sklearn'"

**Solution:** Make sure you activated the environment before running scripts.

```bash
# Conda:
conda activate cs229-quantml

# pip:
source cs229_env/bin/activate  # macOS/Linux
cs229_env\Scripts\activate     # Windows
```

### Issue: "yfinance" not found or connection errors

**Solution:** Check your internet connection. Yahoo Finance requires internet access.

```bash
# Test internet connection
pip install --upgrade yfinance

# Verify yfinance works
python -c "import yfinance as yf; print(yf.Ticker('AAPL').info['currentPrice'])"
```

### Issue: Python version too old

**Solution:** Upgrade Python. This project needs Python 3.9+.

```bash
# Check your version
python --version

# If < 3.9, download newer Python from python.org
# Or use Conda to install a specific version:
conda create -n cs229-quantml python=3.11
```

### Issue: "Kernel PCA fit failed" or numerical errors

**Solution:** This usually means scikit-learn is too old. Upgrade it:

```bash
# Conda:
conda install scikit-learn>=1.5.0

# pip:
pip install --upgrade scikit-learn
```

### Issue: Matplotlib plots not showing

**Solution:** You might need a backend. Add this to your Python script:

```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on your system
import matplotlib.pyplot as plt
```

---

## Package Requirements Explained

Here's what each package does in our project:

**Core ML:**
- `numpy` - Numerical arrays and math operations
- `pandas` - Dataframes for handling options data  
- `scikit-learn` - Machine learning models (SVM, Random Forest, Gradient Boosting)
- `scipy` - Scientific computing (used by Kernel PCA)

**Visualization:**
- `matplotlib` - Creating plots and figures
- `seaborn` - Pretty statistical visualizations

**Financial Data:**
- `yfinance` - Fetches real-time options data from Yahoo Finance
- `py_vollib` - Black-Scholes pricing and Greeks calculations

**Utilities:**
- `python-dateutil` - Date parsing for options expiration dates
- `pyyaml` (optional) - Configuration file reading

---

## Development Setup (For Contributors)

If you want to modify the code or contribute to the project, you might want additional tools:

```bash
# Activate your environment first, then:

# Install development tools
pip install jupyter ipython pytest black flake8

# Install project in editable mode
pip install -e .
```

**Development tools:**
- `jupyter` - Interactive notebooks for exploration
- `ipython` - Better Python shell
- `pytest` - Unit testing
- `black` - Code formatting
- `flake8` - Code linting

---

## Updating the Environment

If we add new dependencies or update versions:

**Conda:**
```bash
conda env update -f environment.yml --prune
```

**pip:**
```bash
pip install --upgrade -r requirements.txt
```

---

## Removing the Environment

If you need to start fresh:

**Conda:**
```bash
conda deactivate
conda env remove -n cs229-quantml
```

**pip:**
```bash
deactivate
rm -rf cs229_env  # On Windows: rmdir /s cs229_env
```

---

## Running on Different Systems

### macOS / Linux

Everything should work out of the box. Just follow the setup instructions above.

### Windows

Use Anaconda Prompt or PowerShell (not Command Prompt) for best results.

If you get SSL errors when fetching data:
```bash
pip install --upgrade certifi
```

### Google Colab / Jupyter Notebooks

If running in Colab:
```python
!pip install yfinance py_vollib scikit-learn==1.5.0
```

Note: Colab already has most packages installed, you just need to add the financial ones.

---

## Still Having Issues?

Contact us:
- Juli Huang: julih@stanford.edu
- Jake Cheng: jiajunc4@stanford.edu
- Rupert Lu: rupertlu@stanford.edu

Or check the GitHub issues page: [https://github.com/TheClassicTechno/CS229_QuantML_Project/issues](https://github.com/TheClassicTechno/CS229_QuantML_Project/issues)

---

**Last Updated:** December 5, 2025
