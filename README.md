# retail-sales-prediction

Based on Kaggle competition in 2015 "Florian Knauer and Will Cukierski. Rossmann Store Sales. https://kaggle.com/competitions/rossmann-store-sales, 2015. Kaggle."

## Executive Summary

This project is aimed to tackle the Kaggle “Rossmann Store Sales” challenge by building an end-to-end machine learning pipeline to forecast six weeks of daily sales for ~1100 retail stores in Germany.
Using a combination of feature engineering, time-based validation, and machine learning models (i.e. Linear Regression, LightGBM, XGBoost), the final solution (using tuned XGBoost model) achieved a pretty strong competitive performance on Kaggle:

🏅 Private Score: 0.12290 RMSPE
🌍 Public Score: 0.12092 RMSPE
🏆 Benchmark: Competition winner achieved approximately 0.10 RMSPE

With the model positioned within ~0.02 RMSPE of the winning solution, this approach may be used to improve sales forecasting to help store managers better planning their inventory, staffing and marketing decisions, i.e.:
- Managers can optimize inventory by directing stocks to avoid stockouts at high-demand stores, and reducing overstocking in low-demand stores
- Also, improve operations - as to make demand-based staffing, scheduling, and replenishment decisions
- Make informed, targeted marketing decisions for stores that need support, and to increase revenue opportunities and prevent unnecessary promotions

## Business Problem

Rossmann, a major European drugstore chain, needs accurate forecasts to manage:

- Inventory planning
- Staff scheduling
- Supply chain operations
- Promotion budgeting

The goal: _Predict daily sales for each store in the test period (6 weeks in advance) based on historical sales and store data_. Accurate forecasts directly reduce operational costs and improve revenue planning.

## Methodology
1. Data Preparation

- Loaded and merged the train, test, and store datasets
- Converted `Date` to datetime format
- Extracted date fields (`DayOfWeek`, `Month`, `Year`)

2. Feature Engineering

The project focused on clean, structured preprocessing using _sklearn preprocessing pipeline_, not heavy time-series transformations. Key features:

Imputed missing values:
- Numerical features → fill NAs with 0, except for `CompetitionDistance` (I assumed NA means competition is very far away, as it doesn't make sense to have distance equals to 0)
- Categorical features → most frequent (SimpleImputer)
- Dropped columns with invariance and excessive missingness
- Standardized the target variable using log1p(`Sales`) to stabilize variance

Scaling: 
- Numerical features kept as-is or scaled using StandardScaler / MinMaxScaler
- Ordinal encoding for `Store` (integer encoding from 1 to 1115)
- One-hot encoding for `PromoInterval`
- Passthrough for binary fields (`Promo`, `Open`, `SchoolHoliday`, `Promo2`)
- Promotion data from store.csv (`Promo2SinceYear`, `Promo2SinceWeek`)
- Competition data used directly without time transformations

3. Modeling

Evaluated three models, first on a time-based validation split, where the last six weeks of the training data served as validation:

- Linear Regression (baseline)
- LightGBM Regressor
- XGBoost Regressor (best overall performance)

Hyperparameter tuning using Optuna was then used to improve performance, especially on LightGBM & XGBoost regressors. The tuned XGBoost model produced the strongest overall results on both validation metrics, and therefore its prediction result on the test set was used to submit to the Kaggle leaderboard score.

4. Evaluation

Metrics used:

- RMSPE (competition metric)
- RMSE (log scale)
- then convert RMSE and RMSPE on original sales scale (using expm1 to reverse the log transform)

5. Results
   on Validation set
   <img width="1189" height="495" alt="image" src="https://github.com/user-attachments/assets/eb44af71-098a-4d24-a663-11e8beac1317" />

With the lowest RMSPE score of 0.124, meaning that the prediction on validation set is not too far off from the actual sales, the **tuned XGBoost model** is hence used to predict sales on test set.

Kaggle scores (by uploading submitted sales prediction per store based on test set):
* Private Leaderboard: 0.12290
* Public Leaderboard: 0.12092

6. Limitations

- The model does not use lag or rolling time-series features, limiting its ability to capture deeper store-level temporal patterns.
- External factors such as weather or local events were not included, which may reduce predictive accuracy.
- Only single boosted-tree models were used; no ensembling or deep time-series models were explored.

7. Future Improvements

- Explore more with lagged sales, rolling averages, and long-term trend features to better capture temporal dynamics.
- Incorporate external data sources (e.g. weather, regional holidays, events) to enrich the feature.
- Experiment with more trials during hyperparameter tuning (the current one used 30 trials), model ensembles or advanced architectures to improve performance beyond individual models.
