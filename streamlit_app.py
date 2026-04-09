import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import plotly.express as px
import time

st.set_page_config(layout="wide", page_title="Real Estate Analytics (Streamlit)")

# Sidebar
st.sidebar.title("Upload & Settings")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

# Helpers
def pricing_algorithm(row):
    base = 200 * row.get("sqft_living", 1000)
    bedrooms_adj = (row.get("bedrooms", 2) - 2) * 5000
    baths_adj = (row.get("bathrooms", 1) - 1) * 4000
    grade_mul = 1 + (row.get("grade", 7) - 7) * 0.03
    waterfront_mul = 1.35 if row.get("waterfront", 0) == 1 else 1.0
    age = 2026 - int(row.get("yr_built", 2000))
    age_dep = max(0.85, 1 - age * 0.003)
    renov_bonus = 1.05 if row.get("yr_renovated", 0) > 0 else 1.0
    price = base + bedrooms_adj + baths_adj
    price *= grade_mul * waterfront_mul * age_dep * renov_bonus
    noise = np.random.uniform(0.97, 1.03)
    return price * noise

def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
    }
    results = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        t = time.time() - start
        results[name] = {"model": model, "time": t}
    return results

# Main layout
st.title("Real Estate Analytics — Streamlit Starter")

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head())

    st.subheader("Data Summary")
    st.write(df.describe(include="all"))

    st.subheader("Missing values")
    miss = df.isna().sum()
    st.bar_chart(miss[miss > 0])

    # Quick feature engineering for demo
    if "price" in df.columns:
        y = df["price"]
        X = df.select_dtypes(include=[np.number]).drop(columns=["price"], errors="ignore").fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        st.subheader("Train models")
        if st.button("Start Training"):
            with st.spinner("Training models..."):
                results = train_models(X_train, y_train)
            metrics = []
            for name, info in results.items():
                pred = info["model"].predict(X_test)
                r2 = r2_score(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                mae = mean_absolute_error(y_test, pred)
                metrics.append((name, r2, rmse, mae, info["time"]))
            dfm = pd.DataFrame(metrics, columns=["Model", "R2", "RMSE", "MAE", "Train time (s)"])
            st.table(dfm.sort_values("R2", ascending=False))

            # Save best model
            best_name = dfm.sort_values("R2", ascending=False).iloc[0]["Model"]
            st.success(f"Best model: {best_name}")
            joblib.dump(results[best_name]["model"], "best_model.joblib")

        st.subheader("Price Prediction (formula demo)")
        sample = {}
        cols = st.columns(4)
        sample["sqft_living"] = cols[0].number_input("sqft_living", value=1500)
        sample["bedrooms"] = cols[1].number_input("bedrooms", value=3)
        sample["bathrooms"] = cols[2].number_input("bathrooms", value=2)
        sample["grade"] = cols[3].slider("grade", 1, 13, 7)
        sample["waterfront"] = st.selectbox("waterfront", [0, 1])
        sample["yr_built"] = st.number_input("yr_built", value=1990)

        if st.button("Predict Price (formula)"):
            pred = pricing_algorithm(sample)
            st.metric("Predicted Price", f"${pred:,.0f}")

    else:
        st.info("CSV must include a `price` column for modelling demo.")
else:
    st.info("Upload a housing CSV to get started.")
