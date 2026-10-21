import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Athlete Performance Analytics", page_icon="🏃", layout="wide")
st.markdown("""
<style>
.main {background:#f7f9fc}
[data-testid="stMetric"] {background:white;padding:16px;border-radius:12px;border:1px solid #e6eaf0}
.hero {padding:24px;border-radius:16px;color:white;background:linear-gradient(120deg,#102a43,#176b87);margin-bottom:18px}
</style>
<div class="hero"><h1 style="margin:0">Athlete Performance Analytics</h1>
<p style="margin:8px 0 0">Explore data, evaluate a model, and predict a performance metric.</p></div>
""", unsafe_allow_html=True)

st.sidebar.title("🏃 Athlete Analytics")
page = st.sidebar.radio("Navigation", ["Overview", "Explore Dataset", "Train & Evaluate", "Predict Performance"])
uploaded = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.session_state["source_df"] = df.copy()
        st.session_state["source_name"] = uploaded.name
    except Exception as e:
        st.error(f"Unable to read CSV: {e}")
        st.stop()
elif "source_df" in st.session_state:
    df = st.session_state["source_df"].copy()
else:
    st.info("Upload your athlete-performance CSV using the uploader in the left sidebar.")
    st.markdown("Your CSV should have one row per record and a numeric target column such as `Performance_Metric`.")
    st.stop()

st.sidebar.success(f"Loaded: {st.session_state.get('source_name', 'dataset')}")
st.sidebar.write(f"Rows: **{len(df):,}** · Columns: **{len(df.columns)}**")

if page == "Overview":
    st.subheader("Dataset overview")
    a, b, c, d = st.columns(4)
    a.metric("Records", f"{len(df):,}")
    b.metric("Columns", f"{len(df.columns):,}")
    c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    d.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    st.markdown("### Dataset preview")
    st.dataframe(df.head(15), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.markdown("### Missing values")
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0].head(15)
        if len(missing):
            st.bar_chart(missing)
        else:
            st.success("No missing values found.")
    with right:
        st.markdown("### Column summary")
        summary = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(t) for t in df.dtypes],
            "Missing": [int(df[c].isna().sum()) for c in df.columns],
            "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

elif page == "Explore Dataset":
    st.subheader("Explore dataset")
    st.dataframe(df, use_container_width=True, height=420)
    col_name = st.selectbox("Choose a column", list(df.columns))
    col = df[col_name]
    a, b, c = st.columns(3)
    a.metric("Non-missing", f"{int(col.notna().sum()):,}")
    b.metric("Missing", f"{int(col.isna().sum()):,}")
    c.metric("Unique values", f"{int(col.nunique(dropna=True)):,}")
    if pd.api.types.is_numeric_dtype(col):
        st.write(col.describe().to_frame("Statistics"))
        fig, ax = plt.subplots()
        col.dropna().hist(ax=ax, bins=30)
        ax.set_xlabel(col_name)
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.bar_chart(col.fillna("Missing").astype(str).value_counts().head(20))
    st.download_button("Download dataset copy", df.to_csv(index=False).encode("utf-8"),
                       "athlete_performance_dataset.csv", "text/csv")

elif page == "Train & Evaluate":
    st.subheader("Train and evaluate a regression model")
    st.caption("The app trains a Random Forest model on your uploaded dataset and evaluates it on held-out test data.")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("A numeric target column is required.")
        st.stop()
    suggested = "Performance_Metric" if "Performance_Metric" in numeric_cols else numeric_cols[-1]
    target = st.selectbox("Numeric target to predict", numeric_cols, index=numeric_cols.index(suggested))
    id_cols = [c for c in df.columns if c != target and c.lower() in {"athlete_id", "id", "athlete_name", "name"}]
    excluded = st.multiselect("Exclude identifiers or columns not available at prediction time",
                              [c for c in df.columns if c != target], default=id_cols)
    test_percent = st.slider("Test data (%)", 10, 40, 20, 5)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
    if st.button("Train model", type="primary", use_container_width=True):
        work = df.drop(columns=excluded).dropna(subset=[target]).drop_duplicates().copy()
        X = work.drop(columns=[target]).copy()
        y = pd.to_numeric(work[target], errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        for c in list(X.columns):
            if "date" in c.lower():
                parsed = pd.to_datetime(X[c], errors="coerce")
                if parsed.notna().mean() >= 0.5:
                    X[c + "_year"] = parsed.dt.year
                    X[c + "_month"] = parsed.dt.month
                    X[c + "_day"] = parsed.dt.day
                    X.drop(columns=[c], inplace=True)
        X.dropna(axis=1, how="all", inplace=True)
        if len(X) < 20 or X.shape[1] == 0:
            st.error("Not enough usable records or features to train.")
            st.stop()
        nums = X.select_dtypes(include=np.number).columns.tolist()
        cats = [c for c in X.columns if c not in nums]
        transformers = []
        if nums:
            transformers.append(("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), nums))
        if cats:
            transformers.append(("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cats))
        pipeline = Pipeline([
            ("preprocess", ColumnTransformer(transformers=transformers)),
            ("model", RandomForestRegressor(n_estimators=250, min_samples_leaf=2,
                                            random_state=int(seed), n_jobs=-1))
        ])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_percent / 100, random_state=int(seed))
        with st.spinner("Training and evaluating..."):
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
        scores = {
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "actual": y_test.to_numpy(), "predicted": preds,
            "train_n": len(X_train), "test_n": len(X_test), "target": target
        }
        st.session_state["model"] = pipeline
        st.session_state["feature_columns"] = list(X.columns)
        st.session_state["training_X"] = X
        st.session_state["evaluation"] = scores
    if "evaluation" in st.session_state:
        e = st.session_state["evaluation"]
        st.markdown("### Test-set metrics")
        a, b, c = st.columns(3)
        a.metric("R² score", f"{e['r2']:.4f}")
        b.metric("MAE", f"{e['mae']:.3f}")
        c.metric("RMSE", f"{e['rmse']:.3f}")
        st.caption(f"Training records: {e['train_n']:,} · Test records: {e['test_n']:,}")
        if e["r2"] < 0:
            st.warning("Negative R² means the model performed worse than predicting the test-set mean. Review target quality and feature relationships.")
        elif e["r2"] >= 0.60:
            st.success("R² is at least 0.60 on this split. Validate further with cross-validation.")
        else:
            st.info("The model explains less than 60% of target variation on this test split.")
        fig, ax = plt.subplots()
        ax.scatter(e["actual"], e["predicted"], alpha=0.5)
        low = min(np.min(e["actual"]), np.min(e["predicted"]))
        high = max(np.max(e["actual"]), np.max(e["predicted"]))
        ax.plot([low, high], [low, high], linestyle="--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs predicted")
        st.pyplot(fig)
        plt.close(fig)
        out = pd.DataFrame({"Actual": e["actual"], "Predicted": e["predicted"],
                            "Absolute_Error": np.abs(e["actual"] - e["predicted"])})
        st.download_button("Download test predictions", out.to_csv(index=False).encode("utf-8"),
                           "athlete_test_predictions.csv", "text/csv")
    else:
        st.info("Choose the target, then click Train model.")

elif page == "Predict Performance":
    st.subheader("Predict performance")
    if "model" not in st.session_state:
        st.warning("Go to Train & Evaluate and train a model first.")
        st.stop()
    model = st.session_state["model"]
    features = st.session_state["feature_columns"]
    training_X = st.session_state["training_X"]
    st.write(f"Target: **{st.session_state['evaluation']['target']}**")
    st.caption("Defaults are based on the training data median or most common category.")
    with st.form("prediction_form"):
        values = {}
        columns = st.columns(2)
        for i, name in enumerate(features):
            series = training_X[name]
            with columns[i % 2]:
                if pd.api.types.is_numeric_dtype(series):
                    default = series.median()
                    if pd.isna(default):
                        default = 0.0
                    values[name] = st.number_input(name.replace("_", " "), value=float(default))
                else:
                    opts = series.dropna().astype(str).value_counts().index.tolist()
                    if not opts:
                        opts = ["Unknown"]
                    values[name] = st.selectbox(name.replace("_", " "), opts)
        submit = st.form_submit_button("Predict performance", type="primary", use_container_width=True)
    if submit:
        pred = float(model.predict(pd.DataFrame([values], columns=features))[0])
        st.markdown("### Prediction result")
        st.metric("Predicted performance metric", f"{pred:.2f}")
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append({"Prediction #": len(st.session_state["history"]) + 1,
                                            "Predicted value": pred})
    if st.session_state.get("history"):
        st.markdown("### This-session prediction history")
        hist = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist, use_container_width=True, hide_index=True)
        st.download_button("Download prediction history", hist.to_csv(index=False).encode("utf-8"),
                           "athlete_prediction_history.csv", "text/csv")

st.sidebar.divider()
st.sidebar.caption("Educational project. Predictions are estimates, not medical or coaching advice.")
