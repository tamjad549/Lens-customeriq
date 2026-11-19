# Lens CustomerIQ

Customer segments and churn prediction for shops (DPO-100, AI/ML internship project).

Takes a shop's sales history and answers two questions: **who are my best customers?** and **who is about to stop buying?**
## Live demo

- Dashboard (Streamlit): https://lens-customeriq-pyrjhbwylabxokxm3tukwr.streamlit.app
- Hugging Face Space (embeds the dashboard): https://huggingface.co/spaces/tamjad549/lens-customeriq
- API: runs locally, see "Run the API" below. The free Hugging Face plan no longer runs Python Spaces, so the dashboard is hosted on Streamlit Community Cloud.
## What it does

- Cleans the sales data and builds RFM features (Recency, Frequency, Monetary) for every customer
- Groups customers into 4 segments with K-Means: Champions, Loyal, At Risk, Lost
- Predicts the chance that each customer buys nothing in the next 90 days (churn risk)
- Explains every score with SHAP
- Streamlit dashboard (upload a CSV, see segments and a call list with reasons) and a FastAPI endpoint (`POST /predict`)

## Data

Online Retail II (UCI Machine Learning Repository, public dataset).

- 1,067,371 raw rows, 777,912 rows after cleaning, 5,840 customers
- The data is **not included** in this repository. Download it and save it as `data/raw/online_retail_II.csv`

## Results

Churn label: no purchase in the 90 days after the cut-off date (10 Sep 2011). 56.5% of customers churned.

5-fold cross-validation:

| Model | Features | Mean ROC-AUC |
|---|---|---|
| Logistic Regression (baseline) | 3 (RFM) | 0.790 |
| Logistic Regression | 7 | 0.795 |
| XGBoost | 3 (RFM) | 0.796 |
| **XGBoost (final)** | **7** | **0.803** |

On the held-out test data the final model scored ROC-AUC 0.806, F1 0.78 and recall 0.804 on churners.

The gain over the baseline is modest. Most of the churn signal is already in Recency, Frequency and Monetary. Class weights were tested and not used, because they lowered the recall.

| Segment | Customers | Median Recency (days) | Median Frequency | Median Monetary |
|---|---|---|---|---|
| Champions | 1,189 | 17 | 13 | 4,909.7 |
| Loyal | 1,254 | 24 | 3 | 725.8 |
| At Risk | 1,460 | 185 | 4 | 1,419.6 |
| Lost | 1,937 | 402 | 1 | 266.4 |

## Project structure

```
data/raw/        original data (not in the repo)
data/processed/  cleaned data and features (not in the repo)
notebooks/       01_eda, 02_cleaning, 03_rfm, 04_churn_data, 05_churn_model, 06_pipeline_test
src/             features.py (cleaning, features, segments, churn) and explain.py (reasons)
models/          saved models
api/main.py      FastAPI
app/demo.py      Streamlit dashboard
reports/         SHAP charts
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the dashboard

```
streamlit run app/demo.py
```

Upload a CSV with these columns: Invoice, StockCode, Quantity, InvoiceDate, Price, Customer ID.

## Run the API

```
python -m uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000/docs and try `POST /predict` with:

```json
{
  "Recency": 443,
  "Frequency": 1,
  "Monetary": 2803.2,
  "Tenure": 443,
  "UniqueProducts": 1,
  "Orders_Last90": 0,
  "AvgOrderValue": 2803.2
}
```

## Notes and limits

- SHAP shows what the model uses to decide, not what causes customers to leave.
- Uploaded files are not saved anywhere, and no real customer data is stored in this repository.
- Only transaction data is used, so the model cannot see things like prices, promotions or competitors.

## Author

M. Taha Ali, AI/ML internship.