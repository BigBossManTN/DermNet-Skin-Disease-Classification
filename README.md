# Car Price Prediction Using Machine Learning

This project builds an end-to-end regression system that predicts the price of used cars based on technical specifications, categorical information, and market features.

---

## Technologies Used

| Technology          | Purpose                         |
| :------------------ | :------------------------------ |
| Python              | Core development language       |
| Pandas, NumPy       | Data cleaning and preprocessing |
| Scikit-Learn        | Regression models and pipelines |
| XGBoost             | Advanced optimized model        |
| Matplotlib, Seaborn | Data visualization              |
| GridSearchCV        | Hyperparameter tuning           |

---

## Dataset Overview

- 205 used cars  
- 26 independent features  
- Mix of numeric and categorical variables  
- Target variable: price (continuous)

### Feature categories include:
- Technical specs (horsepower, engine size, compression ratio)  
- Categorical attributes (brand, body type, fuel type, drive wheel)  
- Market characteristics (manufacturer class, aspiration type)

---

## Project Structure

(Indented instead of fenced to avoid breaking the block)

    car_price_prediction
    ├── data
    │   ├── raw
    │   └── processed
    ├── notebooks
    │   ├── 01_data_overview.ipynb
    │   ├── 02_eda.ipynb
    │   ├── 03_baseline_models.ipynb
    │   └── 04_optimized_models.ipynb
    ├── models
    │   └── best_random_forest.pkl
    ├── docs
    │   └── data_dictionary.txt
    ├── reports
    │   ├── figures
    │   └── presentation.pdf
    └── README.md

---

## Features

- Complete EDA workflow  
- Preprocessing pipeline with scaling and one-hot encoding  
- Baseline models: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest  
- Optimized models: XGBoost, Gradient Boosting, SVR, KNN  
- Evaluation using MAE, RMSE, and R²  
- Final optimized model stored in the models directory  

---

## How to Run Locally

### 1. Install dependencies

    pip install -r requirements.txt

### 2. Open notebooks

Run the .ipynb files inside the notebooks folder with Jupyter or VS Code.

---

## Final Model Performance

Best Model: Optimized Random Forest

- R² ≈ 0.95  
- RMSE ≈ 1939  
- MAE ≈ 1380  

---

## Observations

- Small dataset, so preprocessing had strong impact  
- Outliers affected linear models more than tree-based ones  
- Random Forest handled non-linear patterns effectively  
- Categorical encoding significantly improved prediction quality  

---

## Final Recommendations

| Goal                            | Best Model                    |
| ------------------------------ | ----------------------------- |
| Highest accuracy               | Optimized Random Forest        |
| Balanced performance           | Gradient Boosting / XGBoost    |
| Fast and simple baseline       | Linear Regression              |

---

## References & Credits

- Dataset: Kaggle — Car Price Prediction Dataset  
- Libraries: Scikit-Learn, XGBoost  
- Visualization: Matplotlib, Seaborn  
