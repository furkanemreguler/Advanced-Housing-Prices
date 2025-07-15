# House Prices: Advanced Regression Techniques

![Kaggle](https://img.shields.io/badge/Kaggle-Advanced%20Regression%20Techniques-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Directory Structure](#directory-structure)  
- [Setup & Installation](#setup--installation)  
- [Exploratory Data Analysis](#exploratory-data-analysis)  
- [Feature Engineering & Preprocessing](#feature-engineering--preprocessing)  
- [Modeling](#modeling)  
- [Evaluation](#evaluation)  
- [Submission](#submission)  
- [Results & Leaderboard](#results--leaderboard)  
- [Next Steps](#next-steps)  
- [License](#license)  

---

## 🚀 Project Overview

Predict the final sale price of residential homes in Ames, Iowa, using a variety of regression techniques. This notebook‐driven pipeline walks through:

1. **Data loading & cleaning**  
2. **Exploratory Data Analysis (EDA)**  
3. **Feature engineering & handling missing values**  
4. **Model training (Linear Regression, Random Forest, XGBoost, …)**  
5. **Hyperparameter tuning & stacking**  
6. **Submission file generation**  

---

## 📂 Dataset

All files live in the `data/` folder:

- `train.csv` – 1,460 observations with SalePrice  
- `test.csv` – 1,459 observations, no SalePrice  
- `data_description.txt` – detailed feature descriptions  
- `sample_submission.csv` – example submission format  

Source: [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

## 📁 Directory Structure

Advanced-Housing-Prices/
├── data/
│ ├── train.csv
│ ├── test.csv
│ ├── sample_submission.csv
│ └── data_description.txt
├── model.ipynb
├── requirements.txt
└── submission.csv


---

## 🚀 Getting Started

1. **Clone this repo**  
   git clone https://github.com/yourusername/Advanced-Housing-Prices.git
   cd Advanced-Housing-Prices
   
2. **Create & activate a virtual environment**
  python3 -m venv .venv
  source .venv/bin/activate

3. **Install dependencies**
  pip install -r requirements.txt

4. **Launch Jupyter Notebook**
  jupyter notebook model.ipynb

## model.ipynb Workflow

1. **Data loading**  
   - Read `train.csv` and `test.csv` from `/data`  
   - Inspect feature distributions & missingness  

2. **Exploratory Data Analysis (EDA)**  
   - Target distribution (`SalePrice`)  
   - Correlations, outliers, key plots  

3. **Preprocessing & Feature Engineering**  
   - Drop high‑missing columns  
   - Impute numerics (median) & categoricals (mode)  
   - Log‐transform skewed features  
   - One‑hot encode / label‑encode  

4. **Model Training & Evaluation**  
   - Train/Test split on train set  
   - Compare models:  
     - Linear Regression  
     - Random Forest  
     - XGBoost (`XGBRegressor`)  
   - Use cross‑validation RMSE  

5. **Generate Submission**  
   - Predict on `test.csv`  
   - Write to `submission.csv`  

---

## ⚙️ Dependencies

All required libraries are pinned in `requirements.txt`. Key packages include:

- `pandas`, `numpy`  
- `scikit-learn`  
- `xgboost`  
- `matplotlib`, `seaborn`  
- `jupyter`

To add new packages, install via:
```bash
pip install <package_name>
pip freeze > requirements.txt
'''

   
