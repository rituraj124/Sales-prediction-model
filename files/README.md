# BigMart Sales Prediction

End-to-end regression project predicting `Item_Outlet_Sales` for the BigMart retail dataset (8,523 training rows, 5,681 test rows, 11 raw features).

## What this project demonstrates
- **Data cleaning:** fixed inconsistent category labels (`low fat`/`LF`/`reg`), imputed missing `Item_Weight` (17%) and `Outlet_Size` (28%) using domain-informed logic rather than naive mean/mode fills, and corrected physically-impossible zero values in `Item_Visibility`.
- **Feature engineering:** derived `Item_Category` from ID prefixes, `Outlet_Age`, and `Price_Per_Weight`.
- **Modeling:** trained and compared 4 regressors (Linear, Ridge, Random Forest, Gradient Boosting) on a held-out validation split.
- **Evaluation:** RMSE, MAE, R² — best model is **Random Forest** (RMSE ≈ 1,019, R² ≈ 0.62).
- **Interpretation:** feature importance analysis to explain *why* the model predicts what it does, translated into business recommendations.
- **Deliverables:** trained model file (`bigmart_sales_model.pkl`) and test-set predictions (`submission.csv`).

## Files
- `BigMart_Sales_Prediction.ipynb` — full notebook with analysis, code, and outputs
- `bigmart_sales_model.pkl` — trained Random Forest model (joblib)
- `submission.csv` — predictions for the 5,681 test rows

## How to talk about this with a recruiter
> "I worked with a real messy retail dataset — about 30% of the store-size data was missing and the categories had inconsistent labels. I built a cleaning pipeline that imputed values using domain logic instead of just filling in the mean, engineered a few new features, and compared four regression models. My best model explained about 62% of the variance in sales, and I used feature importance to find that pricing and store format were by far the biggest drivers — which I translated into a few concrete business recommendations."

## Key insight
Price (`Item_MRP`) and outlet format (`Outlet_Type`) dominate sales prediction — together they account for the large majority of the model's predictive signal, while item-level attributes like weight and shelf visibility matter comparatively little.
