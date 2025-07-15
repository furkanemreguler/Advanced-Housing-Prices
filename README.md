# House Prices: Advanced Regression Techniques

![Kaggle](https://img.shields.io/badge/Kaggle-Advanced%20Regression%20Techniques-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents

- Project Overview
- Dataset
- Directory Structure
- Setup & Installation
- Exploratory Data Analysis
- Feature Engineering & Preprocessing
- Modeling
- Evaluation
- Submission
- Results & Leaderboard
- Next Steps
- License

---

## Project Overview

Predict the final sale price of residential homes in Ames, Iowa, using a variety of regression techniques. This notebook‑driven pipeline walks through:

1. Data loading & cleaning  
2. Exploratory Data Analysis (EDA)  
3. Feature engineering & handling missing values  
4. Model training (Linear Regression, Random Forest, XGBoost, etc.)  
5. Hyperparameter tuning & stacking  
6. Submission file generation  

---

## Dataset

All files live in the `data/` folder:

- `train.csv` – 1,460 observations with SalePrice  
- `test.csv` – 1,459 observations, no SalePrice  
- `data_description.txt` – detailed feature descriptions  
- `sample_submission.csv` – example submission format  

Source: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

---

## Directory Structure

Advanced-Housing-Prices/  
├── data/  
│   ├── train.csv  
│   ├── test.csv  
│   ├── sample_submission.csv  
│   └── data_description.txt  
├── model.ipynb  
├── requirements.txt  
└── submission.csv  

---

## Setup & Installation

1. Clone this repo  
   ```
   git clone https://github.com/yourusername/Advanced-Housing-Prices.git
   cd Advanced-Housing-Prices
   ```
2. Create & activate a virtual environment  
   ```
   python3 -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   .venv\Scripts\activate.bat   # Windows (cmd)
   ```
3. Install dependencies  
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Launch Jupyter Notebook  
   ```
   jupyter notebook model.ipynb
   ```

---

## Exploratory Data Analysis

- Visualize the distribution of the target (`SalePrice`), features, and missingness  
- Correlation heatmaps to spot relationships  
- Identify outliers and skewed features  

---

## Feature Engineering & Preprocessing

1. **Missing value handling**:  
   - Numerical features → median imputation  
   - Categorical features → mode imputation  
   - **After these steps, there were _zero_ missing values in both training and test datasets**  
2. **Skew correction** (log-transform skewed features)  
3. **Encoding** (one-hot / label encode categorical variables)  
4. **Feature creation** (e.g. total porch area, age buckets, feature interactions)

All steps are orchestrated via scikit-learn pipelines and custom transformers in `model.ipynb`.

---

## Modeling

We compare multiple regressors:

- Linear Regression  
- Random Forest Regressor  
- XGBoost Regressor (XGBRegressor)  

Example snippet:
```
from sklearn.model_selection import train_test_split
from sklearn.linear_model     import LinearRegression
from sklearn.ensemble         import RandomForestRegressor
from xgboost                  import XGBRegressor
from sklearn.metrics          import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

models = {
    "Linear": LinearRegression(),
    "RF": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGB": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    print(f"{name} RMSE: {rmse:.4f}")
```

---

## Evaluation

- Train/Validation split metrics: RMSE, R²  
- Cross-validation (5-fold)  
- Feature importance for tree-based models  

---

## Submission

Generate predictions on the test set:
```
test_preds = best_model.predict(X_test_processed)
submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": test_preds
})
submission.to_csv("submission.csv", index=False)
```
The final `submission.csv` contained no missing values.

---

## Results & Leaderboard

Model           | Validation RMSE  
--------------- | ----------------  
Linear          | 0.1352           
Random Forest   | 0.1247           
XGBoost         | 0.1184           
Ensemble        | 0.1153           

Final Kaggle Score: 0.13557  

---

## Next Steps

- Add more feature interactions  
- Try LightGBM, CatBoost  
- Bayesian hyperparameter tuning (Optuna)  
- Deploy via REST API  

---

## License

This project is licensed under the MIT License.
