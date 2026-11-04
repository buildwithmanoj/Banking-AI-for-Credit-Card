# Smart Spending Advisor

A behavior-aware payment and card recommendation engine. The app reads how a
customer actually spends (category mix, ticket size, channel, payment loyalty)
and recommends the right card product, with explainable reasoning for both
business and technical reviewers.

Built with XGBoost, CatBoost and Logistic Regression in a soft-vote ensemble,
wrapped in a Streamlit UI.

## What it does

* Ingests transaction-level customer data (10k rows, 200 customers in the
  bundled sample, 2 years of history).
* Aggregates to one row per customer: ticket statistics, category-share
  features, channel-share features, and payment loyalty.
* Trains three models on a 75/25 stratified split and soft-votes the output.
* Maps the predicted primary payment rail to a card product and renders the
  reasoning.

## Two views

**Business view** — one customer at a time, with a spend donut, channel split,
monthly trend, the recommended card, and a one-line reason a product manager
can read without ML vocabulary.

**Technical view** — dataset quality strip, per-model prediction with class
probabilities, agreement indicator, holdout accuracies, confusion matrices,
and feature importances for the two tree models.

## Data format

Expected CSV columns:

```
Customer ID, Category, Item, Quantity, Price Per Unit,
Total Spent, Payment Method, Location, Transaction Date
```

The bundled `sample_data.csv` ships with the repo so the app runs out of the
box. Users can also upload their own CSV from the sidebar.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Stack

Python 3.11, Streamlit, Plotly, scikit-learn, XGBoost, CatBoost.
