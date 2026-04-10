import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ================= CONFIG =================
st.set_page_config(page_title=" Automatic Data Analysis ", layout="wide")

# ================= SESSION INIT =================
if "df" not in st.session_state:
    st.session_state.df = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "analysis" not in st.session_state:
    st.session_state.analysis = None

# ================= HEADER =================
st.title(" Automatic Data Analysis ")
st.caption("Upload CSV → Analyze → Smart Dashboard")

# ================= FILE UPLOAD =================
file = st.file_uploader("Upload CSV", type=["csv"])

# ================= ANALYZE =================
def analyze_data(df):
    rows, cols = df.shape
    missing = df.isnull().sum().sum()
    quality = 100 * (1 - missing / (rows * cols))
    return {"rows": rows, "cols": cols, "missing": missing, "quality": quality}

if file:
    df = pd.read_csv(file)
    st.session_state.df = df
    st.session_state.analysis = analyze_data(df)
    st.session_state.analysis_done = True

# ================= MAIN DASHBOARD =================
if st.session_state.analysis_done and st.session_state.df is not None:

    df = st.session_state.df
    analysis = st.session_state.analysis

    # ================= FILTER =================
    st.sidebar.header("🔍 Filters")
    for col in df.select_dtypes(include="object").columns[:3]:
        val = st.sidebar.multiselect(col, df[col].unique())
        if val:
            df = df[df[col].isin(val)]

    # ================= CLEAN =================
    if st.sidebar.button("🧹 Clean Data"):
        df = df.drop_duplicates().fillna(method="ffill")
        st.session_state.df = df
        st.success("Cleaned")

    # ================= KPIs =================
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", analysis["rows"])
    c2.metric("Columns", analysis["cols"])
    c3.metric("Quality %", f"{analysis['quality']:.1f}")
    c4.metric("Missing", analysis["missing"])

    # ================= DATA =================
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # ================= CHAT =================
    st.subheader("🤖 Chat with Data")
    q = st.text_input("Ask something")

    if q:
        if "average" in q:
            st.write(df.mean(numeric_only=True))
        elif "top" in q:
            st.write(df.head())
        elif "sum" in q:
            st.write(df.sum(numeric_only=True))
        else:
            st.write("Try: average / sum / top")

    # ================= CHARTS =================
    st.subheader("📊 Charts")

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns

    col1, col2 = st.columns(2)

    if len(num_cols) > 0:
        fig = px.histogram(df, x=num_cols[0])
        fig.update_layout(template="plotly_dark")
        col1.plotly_chart(fig, use_container_width=True)

    if len(cat_cols) > 0:
        vc = df[cat_cols[0]].value_counts().head(10)
        fig = px.pie(values=vc.values, names=vc.index)
        fig.update_layout(template="plotly_dark")
        col2.plotly_chart(fig, use_container_width=True)

    # ================= DONUT =================
    if len(cat_cols) > 0:
        st.subheader("🍩 Donut Chart")
        vc = df[cat_cols[0]].value_counts().head(5)
        fig = px.pie(values=vc.values, names=vc.index, hole=0.5)
        st.plotly_chart(fig, use_container_width=True)

    # ================= INSIGHTS =================
    st.subheader("🧠 Insights")
    st.write("Most frequent:", df.iloc[:, 0].mode()[0])

    # ================= PREDICTION =================
    if len(num_cols) > 0:
        st.subheader("🔮 Prediction")
        st.write("Next value (approx):", df[num_cols[0]].mean())

    # ================= COMPARE =================
    st.subheader("📂 Compare Files")

    file2 = st.file_uploader("Upload second CSV")

    if file2:
        df2 = pd.read_csv(file2)
        st.write("File1 rows:", len(df))
        st.write("File2 rows:", len(df2))

    # ================= EXPORT =================
    st.subheader("📥 Export")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "data.csv")

# ================= RESET =================
if st.button("🔄 Reset"):
    st.session_state.clear()