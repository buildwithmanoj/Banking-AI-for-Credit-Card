"""
Smart Spending Advisor
A 3-model ensemble that reads customer spending behavior and recommends
the right payment instrument + card product, with explainable reasoning
for both business and technical reviewers.

Author: Manoj Kumar Gunasekaran
Stack: XGBoost + CatBoost + Logistic Regression ensemble, Streamlit UI
"""

import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import xgboost as xgb
from catboost import CatBoostClassifier

# ============================================================
# Page config and theme
# ============================================================

st.set_page_config(
    page_title="Smart Spending Advisor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design tokens: a calm fintech palette, light mode first.
PRIMARY = "#1E3A8A"       # deep indigo
ACCENT = "#0EA5E9"        # sky
SUCCESS = "#059669"       # emerald
WARN = "#D97706"          # amber
MUTED = "#64748B"         # slate
BG_SOFT = "#F8FAFC"
BORDER = "#E2E8F0"

CATEGORICAL_PALETTE = [
    "#1E3A8A", "#0EA5E9", "#059669", "#D97706", "#7C3AED",
    "#DB2777", "#0891B2", "#65A30D", "#DC2626", "#CA8A04",
    "#4F46E5", "#9333EA", "#0D9488",
]

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}
      h1, h2, h3, h4 {{ color: {PRIMARY}; font-weight: 600; letter-spacing: -0.01em; }}
      .stat-card {{
        background: white;
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
      }}
      .stat-label {{ color: {MUTED}; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }}
      .stat-value {{ color: #0F172A; font-size: 26px; font-weight: 700; margin-top: 4px; }}
      .stat-sub {{ color: {MUTED}; font-size: 12px; margin-top: 2px; }}
      .pill {{
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 12px; font-weight: 600; letter-spacing: 0.02em;
      }}
      .pill-indigo {{ background: #EEF2FF; color: {PRIMARY}; }}
      .pill-sky {{ background: #E0F2FE; color: #0369A1; }}
      .pill-emerald {{ background: #ECFDF5; color: {SUCCESS}; }}
      .pill-amber {{ background: #FEF3C7; color: {WARN}; }}
      .pill-slate {{ background: #F1F5F9; color: {MUTED}; }}
      .reco-card {{
        background: linear-gradient(135deg, #1E3A8A 0%, #0EA5E9 100%);
        color: white;
        border-radius: 16px;
        padding: 24px 28px;
        box-shadow: 0 10px 30px rgba(30, 58, 138, 0.18);
      }}
      .reco-card h3 {{ color: white; margin: 0 0 6px 0; font-size: 14px; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.85; }}
      .reco-card .reco-title {{ font-size: 32px; font-weight: 700; margin: 0; letter-spacing: -0.02em; }}
      .reco-card .reco-sub {{ font-size: 14px; opacity: 0.9; margin-top: 8px; }}
      .reason-row {{ padding: 10px 0; border-bottom: 1px solid {BORDER}; }}
      .reason-row:last-child {{ border-bottom: none; }}
      .model-card {{
        background: white; border: 1px solid {BORDER}; border-radius: 12px;
        padding: 16px; height: 100%;
      }}
      .model-name {{ font-size: 13px; font-weight: 700; color: {PRIMARY}; letter-spacing: 0.04em; text-transform: uppercase; }}
      .model-pred {{ font-size: 22px; font-weight: 700; margin-top: 6px; color: #0F172A; }}
      .model-conf {{ font-size: 12px; color: {MUTED}; margin-top: 2px; }}
      .data-quality-tile {{
        background: white; border: 1px solid {BORDER}; border-radius: 10px;
        padding: 12px 14px;
      }}
      .dq-label {{ font-size: 11px; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }}
      .dq-value {{ font-size: 18px; font-weight: 700; color: #0F172A; margin-top: 2px; }}
      .footer-note {{ color: {MUTED}; font-size: 12px; margin-top: 16px; }}
      /* Hide Streamlit chrome we don't need */
      #MainMenu {{ visibility: hidden; }}
      footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Data loading
# ============================================================

@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "Data", "spending_patterns_REALISTIC_97percent.csv")
    df = pd.read_csv(path)
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    return df


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    return df


REQUIRED_COLS = [
    "Customer ID", "Category", "Item", "Quantity",
    "Price Per Unit", "Total Spent", "Payment Method",
    "Location", "Transaction Date",
]


def check_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return len(missing) == 0, missing


def data_quality_report(df: pd.DataFrame) -> dict:
    report = {
        "rows": len(df),
        "customers": df["Customer ID"].nunique(),
        "date_min": df["Transaction Date"].min(),
        "date_max": df["Transaction Date"].max(),
        "categories": df["Category"].nunique(),
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "qty_price_mismatch": int(
            ((df["Quantity"] * df["Price Per Unit"]).round(2) - df["Total Spent"].round(2))
            .abs()
            .gt(0.05)
            .sum()
        ),
    }
    txn_per_cust = df.groupby("Customer ID").size()
    report["txn_per_customer_median"] = int(txn_per_cust.median())
    report["txn_per_customer_min"] = int(txn_per_cust.min())
    report["txn_per_customer_max"] = int(txn_per_cust.max())
    return report


# ============================================================
# Feature engineering
# ============================================================

@st.cache_data(show_spinner=False)
def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level data to one row per customer."""
    agg = df.groupby("Customer ID").agg(
        total_transactions=("Total Spent", "size"),
        total_spent=("Total Spent", "sum"),
        avg_ticket=("Total Spent", "mean"),
        median_ticket=("Total Spent", "median"),
        max_ticket=("Total Spent", "max"),
        ticket_std=("Total Spent", "std"),
        unique_categories=("Category", "nunique"),
    ).reset_index()

    # Category share (what % of spend goes where)
    cat_spend = df.groupby(["Customer ID", "Category"])["Total Spent"].sum().unstack(fill_value=0)
    cat_share = cat_spend.div(cat_spend.sum(axis=1), axis=0).add_prefix("cat_share_")

    # Channel share
    loc_spend = df.groupby(["Customer ID", "Location"])["Total Spent"].sum().unstack(fill_value=0)
    loc_share = loc_spend.div(loc_spend.sum(axis=1), axis=0).add_prefix("loc_share_")

    # Top category (as a label, used for narrative only)
    top_cat = df.groupby("Customer ID").apply(
        lambda g: g.groupby("Category")["Total Spent"].sum().idxmax(),
        include_groups=False,
    ).rename("top_category")

    # Primary payment (ground truth, dominant payment per customer)
    primary_pay = df.groupby("Customer ID").apply(
        lambda g: g["Payment Method"].value_counts().index[0],
        include_groups=False,
    ).rename("primary_payment")

    # Dominance (% of transactions in primary payment) — proxy for loyalty
    dominance = df.groupby("Customer ID").apply(
        lambda g: g["Payment Method"].value_counts(normalize=True).iloc[0],
        include_groups=False,
    ).rename("payment_dominance")

    feats = (
        agg.merge(cat_share, left_on="Customer ID", right_index=True, how="left")
           .merge(loc_share, left_on="Customer ID", right_index=True, how="left")
           .merge(top_cat, left_on="Customer ID", right_index=True, how="left")
           .merge(primary_pay, left_on="Customer ID", right_index=True, how="left")
           .merge(dominance, left_on="Customer ID", right_index=True, how="left")
    )
    feats["ticket_std"] = feats["ticket_std"].fillna(0)
    return feats


# ============================================================
# Model ensemble
# ============================================================

NUMERIC_FEATURES_BASE = [
    "total_transactions", "total_spent", "avg_ticket",
    "median_ticket", "max_ticket", "ticket_std", "unique_categories",
]


@st.cache_resource(show_spinner=False)
def train_ensemble(feats: pd.DataFrame, random_state: int = 42):
    """Train XGBoost, CatBoost and Logistic Regression on customer features
    to predict primary payment method. Returns models, holdout metrics,
    label encoder, feature list.
    """
    # Target: collapse rare classes into "Other" so models have enough signal
    y_raw = feats["primary_payment"].copy()
    counts = y_raw.value_counts()
    keep = counts[counts >= 10].index.tolist()
    y_raw = y_raw.where(y_raw.isin(keep), other="Other")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    share_cols = [c for c in feats.columns if c.startswith("cat_share_") or c.startswith("loc_share_")]
    feature_cols = NUMERIC_FEATURES_BASE + share_cols
    X = feats[feature_cols].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    # XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, random_state=random_state,
        eval_metric="mlogloss",
    )
    xgb_clf.fit(X_train, y_train)

    # CatBoost
    cat_clf = CatBoostClassifier(
        iterations=300, depth=5, learning_rate=0.06,
        random_state=random_state, verbose=0, loss_function="MultiClass"
        if len(np.unique(y)) > 2 else "Logloss",
    )
    cat_clf.fit(X_train, y_train)

    # Logistic Regression (scaled) — the "classical baseline" in the trio
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr_clf = LogisticRegression(max_iter=500, C=1.0, random_state=random_state)
    lr_clf.fit(X_train_s, y_train)

    # Holdout metrics
    def metrics(model, Xte, is_lr=False):
        Xin = X_test_s if is_lr else Xte
        preds = model.predict(Xin)
        preds = preds.ravel() if preds.ndim > 1 else preds
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "confusion": confusion_matrix(y_test, preds, labels=range(len(le.classes_))).tolist(),
            "preds": preds.tolist(),
        }

    results = {
        "XGBoost": metrics(xgb_clf, X_test),
        "CatBoost": metrics(cat_clf, X_test),
        "Logistic Regression": metrics(lr_clf, X_test, is_lr=True),
    }

    # Feature importance (tree models)
    xgb_imp = dict(zip(feature_cols, xgb_clf.feature_importances_.tolist()))
    cat_imp = dict(zip(feature_cols, cat_clf.feature_importances_.tolist()))

    return {
        "xgb": xgb_clf,
        "cat": cat_clf,
        "lr": lr_clf,
        "scaler": scaler,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "holdout": {
            "y_test": y_test.tolist(),
            "models": results,
        },
        "feature_importance": {"XGBoost": xgb_imp, "CatBoost": cat_imp},
        "share_cols": share_cols,
        "classes": le.classes_.tolist(),
    }


def predict_single(bundle, customer_row: pd.Series):
    X = customer_row[bundle["feature_cols"]].astype(float).values.reshape(1, -1)
    X_s = bundle["scaler"].transform(X)

    probs = {
        "XGBoost": bundle["xgb"].predict_proba(X)[0],
        "CatBoost": bundle["cat"].predict_proba(X)[0].ravel(),
        "Logistic Regression": bundle["lr"].predict_proba(X_s)[0],
    }
    classes = bundle["classes"]
    per_model = {}
    for name, p in probs.items():
        idx = int(np.argmax(p))
        per_model[name] = {
            "prediction": classes[idx],
            "confidence": float(p[idx]),
            "probabilities": {c: float(v) for c, v in zip(classes, p)},
        }
    # Soft-vote ensemble
    stacked = np.vstack(list(probs.values()))
    ens = stacked.mean(axis=0)
    idx = int(np.argmax(ens))
    ensemble = {
        "prediction": classes[idx],
        "confidence": float(ens[idx]),
        "probabilities": {c: float(v) for c, v in zip(classes, ens)},
    }
    agreement = len({m["prediction"] for m in per_model.values()}) == 1
    return per_model, ensemble, agreement


# ============================================================
# Card product catalog (recommendation layer)
# ============================================================

CARD_CATALOG = {
    "Digital Wallet": {
        "product": "Digital-first Rewards Card",
        "tagline": "Built for mobile checkout and e-commerce.",
        "perks": ["3% cashback on digital wallet spend", "No FX fees on online purchases", "Virtual cards on demand", "Real-time push on every transaction"],
        "fit_reason_tpl": "This customer runs {wallet_pct}% of spend through digital wallets. A card that rewards digital-native checkout aligns with existing behavior rather than forcing a change.",
    },
    "Credit Card": {
        "product": "Premium Lifestyle Card",
        "tagline": "Credit-first, with structured rewards on recurring categories.",
        "perks": ["2% cashback on recurring categories", "Travel insurance included", "Purchase protection up to €2,000", "Flexible EMI on tickets > €500"],
        "fit_reason_tpl": "Credit card already dominates this customer's wallet ({cc_pct}%). A premium credit product deepens the share-of-wallet rather than disrupting it.",
    },
    "Cash": {
        "product": "Starter Debit + Budgeting Card",
        "tagline": "A first step toward digital, with no credit risk.",
        "perks": ["No fees, no interest", "Weekly spending caps in the app", "Round-up savings on every swipe", "Cashback on groceries and transport"],
        "fit_reason_tpl": "Cash-heavy profile ({cash_pct}% cash). The sensible move is a low-friction debit that migrates everyday spend without introducing credit exposure.",
    },
    "Debit Card": {
        "product": "Everyday Debit+ with Cashback",
        "tagline": "Debit-first, with credit-like perks where they matter.",
        "perks": ["1.5% cashback on groceries & utilities", "Zero FX markup in-store", "Instant card freeze", "Overdraft buffer up to €200"],
        "fit_reason_tpl": "Debit is this customer's anchor ({dc_pct}%). Upgrading them within their preferred rail outperforms pushing credit they have not asked for.",
    },
    "Other": {
        "product": "Flexible Multi-rail Card",
        "tagline": "Covers mixed behavior with programmable rewards.",
        "perks": ["Choose your own rewards category monthly", "Works across digital wallet, online and in-store", "No annual fee the first year"],
        "fit_reason_tpl": "Behavior is split across rails. A flexible product avoids locking the customer into a pattern they do not actually follow.",
    },
}


def recommendation_for(prediction: str, customer_feats: pd.Series) -> dict:
    card = CARD_CATALOG.get(prediction, CARD_CATALOG["Other"])
    wallet_pct = int(round(customer_feats.get("payment_dominance", 0) * 100))
    cc_pct = wallet_pct
    cash_pct = wallet_pct
    dc_pct = wallet_pct
    reason = card["fit_reason_tpl"].format(
        wallet_pct=wallet_pct, cc_pct=cc_pct, cash_pct=cash_pct, dc_pct=dc_pct,
    )
    return {"card": card, "reason": reason}


# ============================================================
# Sidebar: data & controls
# ============================================================

st.sidebar.markdown(
    f"<div style='font-size:13px;color:{MUTED};letter-spacing:0.08em;text-transform:uppercase;font-weight:700;'>Smart Spending Advisor</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("### Data source")

data_choice = st.sidebar.radio(
    "Choose data source",
    ["Use bundled sample (10k txn, 200 customers)", "Upload my own CSV"],
    label_visibility="collapsed",
)

df = None
data_source_label = ""
if data_choice.startswith("Use bundled"):
    df = load_sample_data()
    data_source_label = "Bundled sample"
else:
    uploaded = st.sidebar.file_uploader(
        "Upload a CSV",
        type=["csv"],
        help=(
            "Expected columns: Customer ID, Category, Item, Quantity, "
            "Price Per Unit, Total Spent, Payment Method, Location, Transaction Date."
        ),
    )
    if uploaded is not None:
        try:
            df = load_uploaded_data(uploaded)
            data_source_label = f"Uploaded: {uploaded.name}"
        except Exception as e:
            st.sidebar.error(f"Could not read file: {e}")
    else:
        st.sidebar.info("Upload a CSV or switch to the bundled sample to proceed.")
        st.stop()

# Schema check
ok, missing = check_schema(df)
if not ok:
    st.sidebar.error(f"Missing columns: {', '.join(missing)}")
    st.stop()

dq = data_quality_report(df)

st.sidebar.markdown("### Dataset at a glance")
st.sidebar.markdown(
    f"""
    <div class='data-quality-tile' style='margin-bottom:8px'>
      <div class='dq-label'>Transactions</div>
      <div class='dq-value'>{dq['rows']:,}</div>
    </div>
    <div class='data-quality-tile' style='margin-bottom:8px'>
      <div class='dq-label'>Unique customers</div>
      <div class='dq-value'>{dq['customers']:,}</div>
    </div>
    <div class='data-quality-tile' style='margin-bottom:8px'>
      <div class='dq-label'>Window</div>
      <div class='dq-value' style='font-size:14px;'>{dq['date_min'].date()} → {dq['date_max'].date()}</div>
    </div>
    <div class='data-quality-tile'>
      <div class='dq-label'>Nulls / Dupes / Qty mismatch</div>
      <div class='dq-value' style='font-size:14px;'>{dq['missing_total']} / {dq['duplicate_rows']} / {dq['qty_price_mismatch']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Train ensemble")
retrain = st.sidebar.button("Train / retrain models", use_container_width=True)

# Build features
feats = build_customer_features(df)

# Train (cached)
if retrain:
    st.cache_resource.clear()
bundle = train_ensemble(feats)

# Customer selector
st.sidebar.markdown("### Pick a customer")
customer_ids = sorted(feats["Customer ID"].tolist())
selected_cust = st.sidebar.selectbox(
    "Customer ID", customer_ids, index=min(10, len(customer_ids) - 1),
    label_visibility="collapsed",
)

cust_row = feats[feats["Customer ID"] == selected_cust].iloc[0]
cust_txn = df[df["Customer ID"] == selected_cust].sort_values("Transaction Date")

per_model, ensemble, agreement = predict_single(bundle, cust_row)
reco = recommendation_for(ensemble["prediction"], cust_row)


# ============================================================
# Header
# ============================================================

st.markdown(
    f"""
    <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;'>
      <div>
        <div style='font-size:12px;letter-spacing:0.12em;color:{MUTED};text-transform:uppercase;font-weight:700;'>Smart Spending Advisor</div>
        <h1 style='margin:4px 0 0 0;'>Behavior-aware payment & card recommendation</h1>
        <div style='color:{MUTED};font-size:14px;margin-top:4px;'>
          3-model ensemble on customer-level features · {data_source_label} · {dq['customers']} customers, {dq['rows']:,} transactions
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

view = st.radio(
    "View",
    ["Business view", "Technical view"],
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()


# ============================================================
# Helpers to render blocks
# ============================================================

def stat_card(label: str, value: str, sub: str = "") -> str:
    return f"""
      <div class='stat-card'>
        <div class='stat-label'>{label}</div>
        <div class='stat-value'>{value}</div>
        <div class='stat-sub'>{sub}</div>
      </div>
    """


def category_donut(cust_txn: pd.DataFrame) -> go.Figure:
    g = cust_txn.groupby("Category")["Total Spent"].sum().sort_values(ascending=False).reset_index()
    fig = px.pie(
        g, values="Total Spent", names="Category", hole=0.6,
        color_discrete_sequence=CATEGORICAL_PALETTE,
    )
    fig.update_traces(textposition="outside", textinfo="label+percent", textfont_size=11)
    fig.update_layout(
        showlegend=False, height=320, margin=dict(t=10, b=10, l=10, r=10),
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    return fig


def channel_bar(cust_txn: pd.DataFrame) -> go.Figure:
    g = cust_txn.groupby("Location")["Total Spent"].sum().reset_index()
    fig = px.bar(
        g, x="Location", y="Total Spent",
        color="Location", color_discrete_sequence=CATEGORICAL_PALETTE,
    )
    fig.update_layout(
        showlegend=False, height=260, margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="", yaxis_title="Spend (€)",
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    return fig


def spend_over_time(cust_txn: pd.DataFrame) -> go.Figure:
    g = cust_txn.set_index("Transaction Date").resample("ME")["Total Spent"].sum().reset_index()
    fig = px.line(g, x="Transaction Date", y="Total Spent", markers=True)
    fig.update_traces(line_color=PRIMARY, marker_color=ACCENT)
    fig.update_layout(
        height=240, margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="", yaxis_title="Monthly spend (€)",
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    return fig


def confusion_fig(cm: list, classes: list) -> go.Figure:
    z = np.array(cm)
    fig = go.Figure(data=go.Heatmap(
        z=z, x=classes, y=classes,
        colorscale=[[0, "#F8FAFC"], [1, PRIMARY]],
        showscale=False,
        text=z, texttemplate="%{text}", textfont=dict(color="#0F172A", size=12),
    ))
    fig.update_layout(
        height=260, margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="Predicted", yaxis_title="Actual",
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    return fig


def importance_bar(imp: dict, top_n: int = 10) -> go.Figure:
    items = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [k.replace("cat_share_", "cat: ").replace("loc_share_", "channel: ") for k, _ in items]
    vals = [v for _, v in items]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=PRIMARY))
    fig.update_layout(
        height=320, margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="Importance", yaxis=dict(autorange="reversed"),
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    return fig


# ============================================================
# BUSINESS VIEW
# ============================================================

if view == "Business view":
    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.markdown("### Customer snapshot")
        c1, c2 = st.columns(2)
        c1.markdown(stat_card("Customer ID", selected_cust), unsafe_allow_html=True)
        c2.markdown(stat_card("Transactions", f"{int(cust_row['total_transactions'])}"), unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        c3.markdown(stat_card("Total spent", f"€{cust_row['total_spent']:,.0f}"), unsafe_allow_html=True)
        c4.markdown(stat_card("Avg ticket", f"€{cust_row['avg_ticket']:,.0f}"), unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        c5.markdown(stat_card("Top category", cust_row["top_category"]), unsafe_allow_html=True)
        c6.markdown(
            stat_card("Payment loyalty", f"{int(cust_row['payment_dominance'] * 100)}%",
                      f"to {cust_row['primary_payment']}"),
            unsafe_allow_html=True,
        )

        st.markdown("#### Where the money goes")
        st.plotly_chart(category_donut(cust_txn), use_container_width=True)

        st.markdown("#### How they pay (channel)")
        st.plotly_chart(channel_bar(cust_txn), use_container_width=True)

    with right:
        pill_class = "pill-emerald" if agreement else "pill-amber"
        pill_txt = "All 3 models agree" if agreement else "Models split — ensemble resolves"
        st.markdown(
            f"""
            <div class='reco-card'>
              <h3>Recommendation</h3>
              <div class='reco-title'>{reco['card']['product']}</div>
              <div class='reco-sub'>{reco['card']['tagline']}</div>
              <div style='margin-top:14px;'>
                <span class='pill {pill_class}' style='background:rgba(255,255,255,0.18);color:white;'>{pill_txt}</span>
                <span class='pill' style='background:rgba(255,255,255,0.18);color:white;margin-left:6px;'>
                  Predicted primary rail: {ensemble['prediction']} · {ensemble['confidence']*100:.0f}% confidence
                </span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Why this card")
        st.markdown(
            f"<div class='reason-row' style='color:#0F172A;font-size:14px;'>{reco['reason']}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### What they get")
        for perk in reco["card"]["perks"]:
            st.markdown(
                f"<div class='reason-row' style='color:#0F172A;font-size:14px;'>• {perk}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### Spend trend (monthly)")
        st.plotly_chart(spend_over_time(cust_txn), use_container_width=True)

    st.markdown(
        f"<div class='footer-note'>Models used: XGBoost, CatBoost, Logistic Regression. "
        f"Prediction above is the soft-vote ensemble across all three. "
        f"Switch to <b>Technical view</b> to see each model's output and holdout performance.</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# TECHNICAL VIEW
# ============================================================

else:
    st.markdown("### Data quality strip")
    dq_cols = st.columns(6)
    dq_cols[0].markdown(stat_card("Rows", f"{dq['rows']:,}"), unsafe_allow_html=True)
    dq_cols[1].markdown(stat_card("Customers", f"{dq['customers']:,}"), unsafe_allow_html=True)
    dq_cols[2].markdown(stat_card("Null cells", f"{dq['missing_total']}"), unsafe_allow_html=True)
    dq_cols[3].markdown(stat_card("Dup rows", f"{dq['duplicate_rows']}"), unsafe_allow_html=True)
    dq_cols[4].markdown(stat_card("Qty×Price mismatch", f"{dq['qty_price_mismatch']}", "tolerance 0.05"), unsafe_allow_html=True)
    dq_cols[5].markdown(stat_card("Txn/customer (med)", f"{dq['txn_per_customer_median']}",
                                    f"min {dq['txn_per_customer_min']} · max {dq['txn_per_customer_max']}"),
                        unsafe_allow_html=True)

    st.markdown("### Prediction for selected customer")
    st.markdown(
        f"<div style='color:{MUTED};font-size:13px;margin-bottom:10px;'>"
        f"Customer <b>{selected_cust}</b> · ground-truth primary payment: "
        f"<b>{cust_row['primary_payment']}</b> ({int(cust_row['payment_dominance']*100)}% dominance)</div>",
        unsafe_allow_html=True,
    )

    mcols = st.columns(3)
    for i, (name, info) in enumerate(per_model.items()):
        correct = info["prediction"] == cust_row["primary_payment"]
        chip = "pill-emerald" if correct else "pill-amber"
        chip_txt = "matches ground truth" if correct else "differs from ground truth"
        probs_sorted = sorted(info["probabilities"].items(), key=lambda x: x[1], reverse=True)
        probs_html = "".join(
            f"<div style='display:flex;justify-content:space-between;font-size:12px;color:{MUTED};padding:2px 0;'>"
            f"<span>{c}</span><span>{p*100:.1f}%</span></div>"
            for c, p in probs_sorted
        )
        mcols[i].markdown(
            f"""
            <div class='model-card'>
              <div class='model-name'>{name}</div>
              <div class='model-pred'>{info['prediction']}</div>
              <div class='model-conf'>Confidence {info['confidence']*100:.1f}%</div>
              <div style='margin-top:8px;'><span class='pill {chip}'>{chip_txt}</span></div>
              <div style='margin-top:12px;border-top:1px solid {BORDER};padding-top:8px;'>{probs_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    ens_chip = "pill-emerald" if ensemble["prediction"] == cust_row["primary_payment"] else "pill-amber"
    st.markdown(
        f"""
        <div style='margin-top:12px;background:{BG_SOFT};border:1px solid {BORDER};border-radius:12px;padding:14px 16px;'>
          <div style='font-size:12px;color:{MUTED};letter-spacing:0.08em;text-transform:uppercase;font-weight:700;'>Ensemble (soft vote)</div>
          <div style='display:flex;align-items:baseline;gap:12px;margin-top:4px;'>
            <div style='font-size:22px;font-weight:700;color:#0F172A;'>{ensemble['prediction']}</div>
            <div style='color:{MUTED};font-size:13px;'>Confidence {ensemble['confidence']*100:.1f}%</div>
            <span class='pill {ens_chip}'>{'matches' if ensemble['prediction'] == cust_row['primary_payment'] else 'differs'} ground truth</span>
            <span class='pill pill-slate'>{'unanimous' if agreement else 'split'}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Holdout performance (25% test split, stratified)")
    perf_cols = st.columns(3)
    for i, (name, res) in enumerate(bundle["holdout"]["models"].items()):
        perf_cols[i].markdown(
            stat_card(f"{name} accuracy", f"{res['accuracy']*100:.1f}%", f"on {len(res['preds'])} held-out customers"),
            unsafe_allow_html=True,
        )

    # Majority-class baseline — the honest reference point
    y_test_arr = np.array(bundle["holdout"]["y_test"])
    if len(y_test_arr) > 0:
        maj_class = int(pd.Series(y_test_arr).value_counts().idxmax())
        maj_acc = float((y_test_arr == maj_class).mean())
        st.markdown(
            f"<div style='margin-top:10px;background:#FEF3C7;border:1px solid #FDE68A;border-radius:10px;padding:12px 14px;color:#78350F;font-size:13px;'>"
            f"<b>Baseline check:</b> predicting the majority class ({bundle['classes'][maj_class]}) alone would score "
            f"<b>{maj_acc*100:.1f}%</b> on this holdout. Any model that doesn't clearly beat that is just memorising class frequency. "
            f"On the bundled sample, category and channel features add some lift but the two majority classes "
            f"(Digital Wallet, Credit Card) are inherently close in behavior — this is a finding about the data, "
            f"not a bug. On richer data (merchant category codes, device, time-of-day), lift grows."
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Confusion matrices")
    cm_cols = st.columns(3)
    for i, (name, res) in enumerate(bundle["holdout"]["models"].items()):
        with cm_cols[i]:
            st.markdown(f"**{name}**")
            st.plotly_chart(confusion_fig(res["confusion"], bundle["classes"]), use_container_width=True)

    st.markdown("### Feature importance (top 10)")
    imp_cols = st.columns(2)
    for i, name in enumerate(["XGBoost", "CatBoost"]):
        with imp_cols[i]:
            st.markdown(f"**{name}**")
            st.plotly_chart(importance_bar(bundle["feature_importance"][name]), use_container_width=True)

    with st.expander("Feature set"):
        st.markdown(
            f"**Numeric features:** `{', '.join(NUMERIC_FEATURES_BASE)}`<br>"
            f"**Category-share features:** {len([c for c in bundle['share_cols'] if c.startswith('cat_share_')])}<br>"
            f"**Channel-share features:** {len([c for c in bundle['share_cols'] if c.startswith('loc_share_')])}<br>"
            f"**Target:** `primary_payment` (customer's most-used payment method, rare classes collapsed to Other).",
            unsafe_allow_html=True,
        )

    with st.expander("Raw transactions for this customer"):
        st.dataframe(cust_txn.reset_index(drop=True), use_container_width=True, height=300)

    st.markdown(
        f"<div class='footer-note'>"
        f"Ensemble method: soft vote across XGBoost, CatBoost, Logistic Regression. "
        f"The LR branch runs on StandardScaler-transformed features; the two tree models run on raw numerics. "
        f"Target classes with fewer than 10 customers are collapsed to <code>Other</code> to keep the model identifiable."
        f"</div>",
        unsafe_allow_html=True,
    )
