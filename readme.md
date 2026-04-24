# Industrial Copper Modeling

A machine learning web application that predicts copper industry selling prices and lead status (WON/LOST) using regression and classification models, enhanced with Google Gemini AI explanations.

🔗 **Live App:** https://copper-modeling-l5ibpajdvtfttcqzammmlk.streamlit.app/

---

## What It Does

The copper industry deals with noisy, skewed sales data. This app solves two business problems:

1. **Selling Price Prediction** — Regression model predicts the price of a copper transaction based on quantity, dimensions, application, and customer details
2. **Lead Classification** — Classification model predicts whether a sales lead will be WON or LOST
3. **AI Explanation** — Google Gemini explains each prediction in plain English

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Regression Model | XGBoost Regressor |
| Classification Model | ExtraTrees Classifier |
| AI Explanations | Google Gemini 2.0 |
| Data Processing | Pandas, NumPy, Scikit-learn |

---

## Model Performance

| Model | Metric | Score |
|---|---|---|
| Regression (XGBoost) | R² | 0.40 |
| Regression (XGBoost) | MAE | 0.20 |
| Classification (ExtraTrees) | Accuracy | 89.5% |
| Classification (ExtraTrees) | F1 Score | 0.93 |

Note: Regression R² is lower due to inherent noise in copper pricing data — as documented in the problem statement.

---

## Key Data Processing Steps

- Removed garbage `material_ref` values starting with `00000`
- Applied log transformation to `selling_price` and `quantity tons` (skewness reduced from 301 to ~0)
- Applied log transformation to `thickness`
- Label encoded `item type`
- Filtered classification dataset to WON/LOST only (removed Draft, Offered, etc.)

---

## Project Structure

```
copper-modeling/
├── app.py                  # Streamlit application
├── copper_functions.py     # Data processing + model training + prediction + Gemini
├── Copper_Set.xlsx         # Dataset
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Copper theme
└── README.md
```

---

## How to Run Locally

1. Clone the repo:
```
git clone https://github.com/akashravuru/copper-modeling
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Create `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key
```

4. Run:
```
streamlit run app.py
```

Note: Models train on first load (~2 minutes). Subsequent predictions are instant.

---

Built by [Akash Ravuru](https://linkedin.com/in/akashravuru)