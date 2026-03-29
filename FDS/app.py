"""
Interactive Car Price Prediction
An End-To-End Data Science Playground for Automotive Price Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Interactive Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #0e1117; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1a2740 100%);
    border-right: 1px solid #2a3f5f;
}
[data-testid="stSidebar"] * { color: #c9d6e3 !important; }
.banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a3055 50%, #0d1b2a 100%);
    border: 1px solid #2a3f5f; border-radius: 12px;
    padding: 1.5rem 2rem; margin-bottom: 1.5rem;
}
.banner h1 { font-family: 'Space Mono', monospace; font-size: 1.9rem; color: #e8f4ff; margin: 0; }
.banner p  { color: #7eb8f7; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
.banner .tag {
    display: inline-block; background: rgba(100,180,255,0.12);
    border: 1px solid rgba(100,180,255,0.3); border-radius: 20px;
    padding: 2px 10px; font-size: 0.75rem; color: #7eb8f7; margin-right: 6px; margin-top: 8px;
}
.metric-card {
    background: linear-gradient(135deg, #141e2e 0%, #1c2d44 100%);
    border: 1px solid #2a3f5f; border-radius: 10px;
    padding: 1rem 1.2rem; text-align: center;
}
.metric-card .val { font-family: 'Space Mono', monospace; font-size: 1.6rem; color: #64b4ff; font-weight: 700; }
.metric-card .lbl { font-size: 0.75rem; color: #7a9ab8; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }
.section-title {
    font-family: 'Space Mono', monospace; font-size: 1rem; font-weight: 700;
    color: #64b4ff; letter-spacing: 0.05em; text-transform: uppercase;
    padding: 0.5rem 0; border-bottom: 1px solid #2a3f5f; margin-bottom: 1rem;
}
.info-box { background: rgba(100,180,255,0.07); border-left: 3px solid #64b4ff; border-radius: 0 8px 8px 0; padding: 0.7rem 1rem; margin: 0.5rem 0; color: #c9d6e3; font-size: 0.88rem; }
.success-box { background: rgba(0,200,140,0.07); border-left: 3px solid #00c88c; border-radius: 0 8px 8px 0; padding: 0.7rem 1rem; margin: 0.5rem 0; color: #c9d6e3; font-size: 0.88rem; }
.warning-box { background: rgba(255,180,50,0.07); border-left: 3px solid #ffb432; border-radius: 0 8px 8px 0; padding: 0.7rem 1rem; margin: 0.5rem 0; color: #c9d6e3; font-size: 0.88rem; }
.divider { border: none; border-top: 1px solid #2a3f5f; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
PALETTE  = ["#64b4ff","#00c88c","#ffb432","#ff6b8a","#a78bfa","#38bdf8","#fb923c"]
BG_COLOR = "#0e1117"
CARD_CLR = "#141e2e"
TEXT_CLR = "#c9d6e3"

def style_fig(fig, ax_list=None):
    fig.patch.set_facecolor(BG_COLOR)
    for ax in (ax_list or fig.axes):
        ax.set_facecolor(CARD_CLR)
        ax.tick_params(colors=TEXT_CLR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(TEXT_CLR)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a3f5f")
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file=None):
    if file:
        return pd.read_csv(file)
    return pd.read_csv("car_price_prediction_.csv")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚗 Interactive Car Price Prediction")
    st.markdown("<hr style='border-color:#2a3f5f;margin:0.5rem 0'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("<hr style='border-color:#2a3f5f;margin:0.8rem 0'>", unsafe_allow_html=True)
    section = st.radio("Navigation", [
        "📊 Dataset Overview",
        "🧹 Data Cleaning",
        "🔍 Dynamic Filtering",
        "📈 EDA & Visualisations",
        "🤖 ML Modelling",
    ], label_visibility="collapsed")
    st.markdown("<hr style='border-color:#2a3f5f;margin:0.8rem 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;color:#4a6a8a;text-align:center'>Interactive Car Price Prediction v1.0<br>Car Price Prediction Dataset<br>2500 Records · 10 Features</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
raw_df   = load_data(uploaded)
NUM_COLS = raw_df.select_dtypes(include=[np.number]).columns.tolist()
CAT_COLS = raw_df.select_dtypes(include=["object","string"]).columns.tolist()
ALL_COLS = raw_df.columns.tolist()

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='banner'>
  <h1>🚗 Interactive Car Price Prediction</h1>
  <p>An End-To-End Interactive Data Science Playground · Car Price Prediction Dataset</p>
  <span class='tag'>📁 CSV Upload</span>
  <span class='tag'>🔍 EDA</span>
  <span class='tag'>🧹 Cleaning</span>
  <span class='tag'>📊 Visualisations</span>
  <span class='tag'>🤖 ML Models</span>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1 — DATASET OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if section == "📊 Dataset Overview":
    st.markdown("<div class='section-title'>📊 Dataset Overview</div>", unsafe_allow_html=True)

    rows, cols = raw_df.shape
    missing = int(raw_df.isnull().sum().sum())
    dupes   = int(raw_df.duplicated().sum())

    c1,c2,c3,c4,c5 = st.columns(5)
    for col_obj, val, lbl in [
        (c1, f"{rows:,}", "Total Rows"),
        (c2, cols,         "Columns"),
        (c3, missing,      "Missing Values"),
        (c4, dupes,        "Duplicate Rows"),
        (c5, f"{len(NUM_COLS)}N / {len(CAT_COLS)}C", "Num / Cat Cols"),
    ]:
        col_obj.markdown(f"<div class='metric-card'><div class='val'>{val}</div><div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    c_left, c_right = st.columns([2,1])
    with c_left:
        st.markdown("**Dataset Preview (first 10 rows)**")
        st.dataframe(raw_df.head(10), use_container_width=True)
    with c_right:
        st.markdown("**Column Info**")
        dtype_df = pd.DataFrame({
            "Column": ALL_COLS,
            "Type":   [str(raw_df[c].dtype) for c in ALL_COLS],
            "Nulls":  [raw_df[c].isnull().sum() for c in ALL_COLS],
            "Unique": [raw_df[c].nunique() for c in ALL_COLS],
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**Descriptive Statistics**")
    st.dataframe(raw_df[NUM_COLS].describe().T.round(2), use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**Missing Value Map**")
    if missing == 0:
        st.markdown("<div class='success-box'>No missing values detected. The dataset is complete and ready for analysis.</div>", unsafe_allow_html=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 2))
        miss_pct = raw_df.isnull().sum() / len(raw_df) * 100
        ax.bar(miss_pct.index, miss_pct.values, color=PALETTE[0])
        ax.set_ylabel("Missing %")
        ax.set_title("Missing Value % per Column")
        plt.xticks(rotation=45, ha='right')
        style_fig(fig)
        st.pyplot(fig)

    c1, c2 = st.columns(2)
    c1.markdown(f"<div class='info-box'>📐 <strong>Numeric Columns:</strong> {', '.join(NUM_COLS)}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='info-box'>🏷 <strong>Categorical Columns:</strong> {', '.join(CAT_COLS)}</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# 2 — DATA CLEANING
# ═════════════════════════════════════════════════════════════════════════════
elif section == "🧹 Data Cleaning":
    st.markdown("<div class='section-title'>🧹 Data Cleaning</div>", unsafe_allow_html=True)

    df_clean = raw_df.copy()
    cleaning_log = []

    # Duplicates
    st.markdown("#### 1. Duplicate Record Removal")
    dupes_before = int(df_clean.duplicated().sum())
    c1, c2 = st.columns([3,1])
    rm_dupes = c1.checkbox("Remove duplicate rows", value=True)
    if rm_dupes and dupes_before > 0:
        df_clean = df_clean.drop_duplicates()
        cleaning_log.append(f"Removed {dupes_before} duplicate rows.")
        c2.markdown(f"<div class='success-box'>Removed {dupes_before}</div>", unsafe_allow_html=True)
    else:
        cleaning_log.append("No duplicates found." if dupes_before == 0 else "Duplicate removal skipped.")
        c2.markdown(f"<div class='info-box'>0 dupes</div>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Missing values
    st.markdown("#### 2. Missing Value Imputation")
    miss_total = df_clean.isnull().sum().sum()
    if miss_total == 0:
        st.markdown("<div class='success-box'>No missing values — no imputation required.</div>", unsafe_allow_html=True)
        cleaning_log.append("No missing values to impute.")
    else:
        strategy = st.selectbox("Strategy", ["Mean (numeric) + Mode (categorical)", "Median + Mode", "Drop rows with nulls"])
        if st.button("Apply Imputation"):
            for col in df_clean.columns:
                if df_clean[col].isnull().sum() > 0:
                    if df_clean[col].dtype in [np.float64, np.int64]:
                        fill_val = df_clean[col].mean() if "Mean" in strategy else df_clean[col].median()
                        df_clean[col].fillna(fill_val, inplace=True)
                        cleaning_log.append(f"'{col}': filled nulls with {strategy.split()[0].lower()} ({fill_val:.2f})")
                    else:
                        mode_val = df_clean[col].mode()[0]
                        df_clean[col].fillna(mode_val, inplace=True)
                        cleaning_log.append(f"'{col}': filled nulls with mode ('{mode_val}')")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Outlier detection
    st.markdown("#### 3. Outlier Detection (IQR Method)")
    out_col = st.selectbox("Column for outlier check", NUM_COLS)
    Q1 = df_clean[out_col].quantile(0.25)
    Q3 = df_clean[out_col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = df_clean[(df_clean[out_col] < lo) | (df_clean[out_col] > hi)]
    c1,c2,c3 = st.columns(3)
    c1.metric("Q1", f"{Q1:,.1f}")
    c2.metric("Q3", f"{Q3:,.1f}")
    c3.metric("Outliers Found", len(outliers))
    if st.checkbox("Remove outliers for selected column"):
        df_clean = df_clean[(df_clean[out_col] >= lo) & (df_clean[out_col] <= hi)]
        cleaning_log.append(f"Removed {len(outliers)} outliers from '{out_col}'.")
        st.markdown(f"<div class='success-box'>Removed {len(outliers)} outliers from '{out_col}'.</div>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Log + updated metrics
    st.markdown("#### 4. Cleaning Log")
    for item in cleaning_log:
        st.markdown(f"<div class='success-box'>✅ {item}</div>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**Updated Dataset Metrics**")
    rows2, cols2 = df_clean.shape
    miss2 = int(df_clean.isnull().sum().sum())
    c1,c2,c3 = st.columns(3)
    for col_obj, val, lbl in [(c1, f"{rows2:,}","Rows After Cleaning"),(c2,cols2,"Columns"),(c3,miss2,"Remaining Nulls")]:
        col_obj.markdown(f"<div class='metric-card'><div class='val'>{val}</div><div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df_clean.head(10), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# 3 — DYNAMIC FILTERING
# ═════════════════════════════════════════════════════════════════════════════
elif section == "🔍 Dynamic Filtering":
    st.markdown("<div class='section-title'>🔍 Dynamic Filtering</div>", unsafe_allow_html=True)
    df_f = raw_df.copy()

    c1, c2 = st.columns(2)
    brand_sel = c1.multiselect("Brand",        sorted(df_f['Brand'].unique()),       default=list(df_f['Brand'].unique()))
    fuel_sel  = c2.multiselect("Fuel Type",    sorted(df_f['Fuel Type'].unique()),   default=list(df_f['Fuel Type'].unique()))
    c3, c4    = st.columns(2)
    trans_sel = c3.multiselect("Transmission", sorted(df_f['Transmission'].unique()),default=list(df_f['Transmission'].unique()))
    cond_sel  = c4.multiselect("Condition",    sorted(df_f['Condition'].unique()),   default=list(df_f['Condition'].unique()))

    yr_min, yr_max = int(df_f['Year'].min()), int(df_f['Year'].max())
    year_range  = st.slider("Year Range",  yr_min, yr_max, (yr_min, yr_max))
    pr_min, pr_max = float(df_f['Price'].min()), float(df_f['Price'].max())
    price_range = st.slider("Price Range", pr_min, pr_max, (pr_min, pr_max), step=500.0)

    mask = (
        df_f['Brand'].isin(brand_sel) &
        df_f['Fuel Type'].isin(fuel_sel) &
        df_f['Transmission'].isin(trans_sel) &
        df_f['Condition'].isin(cond_sel) &
        df_f['Year'].between(*year_range) &
        df_f['Price'].between(*price_range)
    )
    filtered = df_f[mask]
    removed  = len(df_f) - len(filtered)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col_obj, val, lbl in [
        (c1, f"{len(filtered):,}", "Rows Retained"),
        (c2, f"{removed:,}",       "Rows Removed"),
        (c3, f"{len(filtered)/len(df_f)*100:.1f}%", "Data Retained"),
    ]:
        col_obj.markdown(f"<div class='metric-card'><div class='val'>{val}</div><div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)

    if removed > 0:
        st.markdown(f"<div class='warning-box'>{removed} rows removed. {len(filtered)} rows match the current filters.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='success-box'>All rows match the current filter criteria.</div>", unsafe_allow_html=True)

    st.markdown("**Filtered Data Preview**")
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# 4 — EDA & VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════
elif section == "📈 EDA & Visualisations":
    st.markdown("<div class='section-title'>📈 EDA & Visualisations</div>", unsafe_allow_html=True)
    df_eda = raw_df.copy()

    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
        "🔥 Correlation Heatmap","📉 Histograms & KDE","🔵 Scatter Plots",
        "📊 Bar Charts","📦 Box Plots","📅 Line Charts",
    ])

    # ── Heatmap ─────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("**Correlation Matrix — Numeric Features**")
        corr = df_eda[NUM_COLS].corr()
        fig, ax = plt.subplots(figsize=(7,5))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask,
                    ax=ax, linewidths=0.5, annot_kws={"size":10,"color":"white"})
        ax.set_title("Correlation Heatmap", color=TEXT_CLR, pad=12)
        style_fig(fig)
        st.pyplot(fig)
        st.markdown("<div class='info-box'>💡 <strong>Insight:</strong> Year and Price show a moderate positive correlation — newer cars command higher prices. Mileage has a negative correlation with Price, confirming depreciation with usage. Engine Size moderately correlates with Price.</div>", unsafe_allow_html=True)

    # ── Histograms ──────────────────────────────────────────────────────────
    with tab2:
        hist_col = st.selectbox("Column", NUM_COLS, key="hist")
        fig, ax = plt.subplots(figsize=(8,4))
        data = df_eda[hist_col].dropna()
        ax.hist(data, bins=40, color=PALETTE[0], edgecolor=BG_COLOR, alpha=0.7, density=True)
        data.plot.kde(ax=ax, color=PALETTE[2], linewidth=2.5, label="KDE")
        ax.set_title(f"Distribution of {hist_col}", color=TEXT_CLR)
        ax.set_xlabel(hist_col); ax.set_ylabel("Density")
        ax.legend(facecolor=CARD_CLR, edgecolor="#2a3f5f", labelcolor=TEXT_CLR)
        style_fig(fig)
        st.pyplot(fig)
        s = data.describe()
        c1,c2,c3,c4 = st.columns(4)
        for col_obj,key in [(c1,"mean"),(c2,"std"),(c3,"min"),(c4,"max")]:
            col_obj.markdown(f"<div class='metric-card'><div class='val'>{s[key]:,.1f}</div><div class='lbl'>{key.title()}</div></div>", unsafe_allow_html=True)
        cv = s["std"]/s["mean"]
        st.markdown(f"<div class='info-box'>💡 <strong>Insight:</strong> {hist_col} has a mean of {s['mean']:,.1f} with std {s['std']:,.1f}. Coefficient of variation: {cv:.2f} — {'high' if cv>0.5 else 'moderate'} variability. The KDE curve shows the smooth probability density.</div>", unsafe_allow_html=True)

    # ── Scatter ─────────────────────────────────────────────────────────────
    with tab3:
        c1,c2,c3 = st.columns(3)
        x_col = c1.selectbox("X axis", NUM_COLS, index=NUM_COLS.index("Mileage") if "Mileage" in NUM_COLS else 0, key="sx")
        y_col = c2.selectbox("Y axis", NUM_COLS, index=NUM_COLS.index("Price") if "Price" in NUM_COLS else 1, key="sy")
        hue   = c3.selectbox("Colour by", ["None"]+CAT_COLS, key="shue")
        fig, ax = plt.subplots(figsize=(9,5))
        if hue == "None":
            ax.scatter(df_eda[x_col], df_eda[y_col], alpha=0.35, color=PALETTE[0], s=12)
        else:
            for i, cat in enumerate(sorted(df_eda[hue].unique())):
                sub = df_eda[df_eda[hue]==cat]
                ax.scatter(sub[x_col], sub[y_col], alpha=0.45, color=PALETTE[i%len(PALETTE)], s=12, label=cat)
            ax.legend(facecolor=CARD_CLR, edgecolor="#2a3f5f", labelcolor=TEXT_CLR, fontsize=8)
        z  = np.polyfit(df_eda[x_col], df_eda[y_col], 1)
        xs = np.linspace(df_eda[x_col].min(), df_eda[x_col].max(), 200)
        ax.plot(xs, np.poly1d(z)(xs), color=PALETTE[2], linewidth=2, linestyle="--", label="Trendline")
        ax.set_xlabel(x_col); ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}", color=TEXT_CLR)
        style_fig(fig); st.pyplot(fig)
        corr_val = df_eda[[x_col,y_col]].corr().iloc[0,1]
        direction = "positive" if corr_val > 0 else "negative"
        strength  = "strong" if abs(corr_val) > 0.6 else ("moderate" if abs(corr_val) > 0.3 else "weak")
        st.markdown(f"<div class='info-box'>💡 <strong>Insight:</strong> Correlation r = <strong>{corr_val:.3f}</strong> — a {strength} {direction} relationship. The dashed trendline (slope = {z[0]:.2f}) confirms the overall direction of association.</div>", unsafe_allow_html=True)

    # ── Bar Charts ──────────────────────────────────────────────────────────
    with tab4:
        bar_cat = st.selectbox("Categorical column", CAT_COLS, key="bcat")
        bar_num = st.selectbox("Numeric column", NUM_COLS, index=NUM_COLS.index("Price") if "Price" in NUM_COLS else 0, key="bnum")
        agg_fn  = st.radio("Aggregation", ["mean","median","sum","count"], horizontal=True)
        if agg_fn == "count":
            bar_data = df_eda[bar_cat].value_counts()
        else:
            bar_data = df_eda.groupby(bar_cat)[bar_num].agg(agg_fn).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9,4))
        ax.bar(bar_data.index.astype(str), bar_data.values,
               color=[PALETTE[i%len(PALETTE)] for i in range(len(bar_data))], edgecolor=BG_COLOR, linewidth=0.5)
        ax.set_xlabel(bar_cat); ax.set_ylabel(f"{agg_fn} {bar_num}" if agg_fn!="count" else "Count")
        ax.set_title(f"{agg_fn.title()} {bar_num} by {bar_cat}", color=TEXT_CLR)
        plt.xticks(rotation=30, ha='right')
        style_fig(fig); st.pyplot(fig)
        top_cat = bar_data.idxmax()
        st.markdown(f"<div class='info-box'>💡 <strong>Insight:</strong> <strong>{top_cat}</strong> leads with {agg_fn} of <strong>{bar_data.max():,.1f}</strong>. This bar chart reveals which {bar_cat} segment dominates in terms of {bar_num}.</div>", unsafe_allow_html=True)

    # ── Box Plots ───────────────────────────────────────────────────────────
    with tab5:
        box_num = st.selectbox("Numeric column", NUM_COLS, index=NUM_COLS.index("Price") if "Price" in NUM_COLS else 0, key="bxn")
        box_cat = st.selectbox("Group by", CAT_COLS, key="bxc")
        labels  = sorted(df_eda[box_cat].unique())
        groups  = [df_eda[df_eda[box_cat]==cat][box_num].dropna().values for cat in labels]
        fig, ax = plt.subplots(figsize=(9,4))
        bp = ax.boxplot(groups, labels=labels, patch_artist=True,
                        medianprops=dict(color=PALETTE[2], linewidth=2))
        for patch, color in zip(bp['boxes'], PALETTE):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        ax.set_xlabel(box_cat); ax.set_ylabel(box_num)
        ax.set_title(f"Distribution of {box_num} by {box_cat}", color=TEXT_CLR)
        plt.xticks(rotation=20)
        style_fig(fig); st.pyplot(fig)
        medians   = {cat: float(np.median(g)) for cat, g in zip(labels, groups) if len(g) > 0}
        top_med   = max(medians, key=medians.get)
        st.markdown(f"<div class='info-box'>💡 <strong>Insight:</strong> <strong>{top_med}</strong> has the highest median {box_num} ({medians[top_med]:,.0f}). Box width shows data spread; whisker extremes and dots reveal outlier distribution per group.</div>", unsafe_allow_html=True)

    # ── Line Charts ─────────────────────────────────────────────────────────
    with tab6:
        line_num = st.selectbox("Numeric column", NUM_COLS, index=NUM_COLS.index("Price") if "Price" in NUM_COLS else 0, key="ln")
        line_agg = st.radio("Aggregation", ["mean","median","sum"], horizontal=True, key="lagg")
        line_hue = st.selectbox("Split by category (optional)", ["None"]+CAT_COLS, key="lhue")
        fig, ax  = plt.subplots(figsize=(10,4))
        if line_hue == "None":
            trend = df_eda.groupby("Year")[line_num].agg(line_agg)
            ax.plot(trend.index, trend.values, color=PALETTE[0], linewidth=2.5, marker='o', markersize=4)
            ax.fill_between(trend.index, trend.values, alpha=0.12, color=PALETTE[0])
        else:
            for i, cat in enumerate(sorted(df_eda[line_hue].unique())):
                sub_trend = df_eda[df_eda[line_hue]==cat].groupby("Year")[line_num].agg(line_agg)
                ax.plot(sub_trend.index, sub_trend.values, color=PALETTE[i%len(PALETTE)],
                        linewidth=2, marker='o', markersize=3, label=cat)
            ax.legend(facecolor=CARD_CLR, edgecolor="#2a3f5f", labelcolor=TEXT_CLR, fontsize=8)
        ax.set_xlabel("Year"); ax.set_ylabel(f"{line_agg.title()} {line_num}")
        ax.set_title(f"{line_agg.title()} {line_num} Over Years", color=TEXT_CLR)
        style_fig(fig); st.pyplot(fig)
        trend_all = df_eda.groupby("Year")[line_num].agg(line_agg)
        pct = (trend_all.iloc[-1] - trend_all.iloc[0]) / trend_all.iloc[0] * 100
        st.markdown(f"<div class='info-box'>💡 <strong>Insight:</strong> {line_num} shows a <strong>{'upward' if pct>0 else 'downward'} trend of {abs(pct):.1f}%</strong> from {int(trend_all.index[0])} to {int(trend_all.index[-1])}. This time-series view reveals long-term market movements in the dataset.</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# 5 — ML MODELLING
# ═════════════════════════════════════════════════════════════════════════════
elif section == "🤖 ML Modelling":
    st.markdown("<div class='section-title'>🤖 Machine Learning Modelling</div>", unsafe_allow_html=True)

    df_ml = raw_df.copy()
    le    = LabelEncoder()
    for col in CAT_COLS:
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))

    model_type = st.radio("Task", ["🏷️ Regression — Predict Price", "🎯 Classification — Budget vs Premium"], horizontal=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    feat_options = [c for c in df_ml.columns if c not in ["Car ID","Price"]]
    selected_features = st.multiselect("Feature Columns (X)", feat_options,
        default=["Year","Engine Size","Fuel Type","Transmission","Mileage","Condition","Brand"])

    c1,c2 = st.columns(2)
    test_size  = c1.slider("Test Size (%)", 10, 40, 20) / 100
    rand_state = c2.slider("Random State", 0, 100, 42)

    if "Regression" in model_type:
        algo = st.selectbox("Algorithm", ["Linear Regression","Ridge Regression","Random Forest","Gradient Boosting"])
    else:
        df_ml["PriceCategory"] = (df_ml["Price"] > df_ml["Price"].median()).astype(int)
        algo = st.selectbox("Algorithm", ["Logistic Regression"])

    if st.button("🚀 Train Model", use_container_width=True):
        if not selected_features:
            st.warning("Select at least one feature column.")
        else:
            X = df_ml[selected_features].values
            y = df_ml["Price"].values if "Regression" in model_type else df_ml["PriceCategory"].values
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=rand_state)
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            if algo=="Linear Regression":   mdl = LinearRegression()
            elif algo=="Ridge Regression":  mdl = Ridge(alpha=1.0)
            elif algo=="Random Forest":      mdl = RandomForestRegressor(n_estimators=100,random_state=rand_state)
            elif algo=="Gradient Boosting": mdl = GradientBoostingRegressor(n_estimators=100,random_state=rand_state)
            else:                            mdl = LogisticRegression(max_iter=1000,random_state=rand_state)

            mdl.fit(X_train, y_train)
            y_pred = mdl.predict(X_test)
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # ── Regression results ──────────────────────────────────────────
            if "Regression" in model_type:
                mse  = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae  = mean_absolute_error(y_test, y_pred)
                r2   = r2_score(y_test, y_pred)

                st.markdown("**Performance Metrics**")
                c1,c2,c3,c4 = st.columns(4)
                for col_obj,val,lbl in [
                    (c1,f"{r2:.4f}","R² Score"),
                    (c2,f"${rmse:,.0f}","RMSE"),
                    (c3,f"${mae:,.0f}","MAE"),
                    (c4,f"{(1-mae/np.mean(np.abs(y_test)))*100:.1f}%","Accuracy Proxy"),
                ]:
                    col_obj.markdown(f"<div class='metric-card'><div class='val'>{val}</div><div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                fig, axes = plt.subplots(1,2,figsize=(12,4))
                axes[0].scatter(y_test, y_pred, alpha=0.35, color=PALETTE[0], s=10)
                axes[0].plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--',lw=2)
                axes[0].set_xlabel("Actual Price"); axes[0].set_ylabel("Predicted Price")
                axes[0].set_title("Actual vs Predicted", color=TEXT_CLR)

                residuals = y_test - y_pred
                axes[1].hist(residuals, bins=40, color=PALETTE[1], edgecolor=BG_COLOR, alpha=0.75)
                axes[1].axvline(0, color=PALETTE[2], linewidth=2, linestyle='--')
                axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Frequency")
                axes[1].set_title("Residual Distribution", color=TEXT_CLR)
                style_fig(fig, axes); st.pyplot(fig)

                if hasattr(mdl, "feature_importances_"):
                    st.markdown("**Feature Importance**")
                    fi = pd.Series(mdl.feature_importances_, index=selected_features).sort_values(ascending=False)
                    fig2, ax2 = plt.subplots(figsize=(8,3))
                    ax2.barh(fi.index, fi.values, color=[PALETTE[i%len(PALETTE)] for i in range(len(fi))])
                    ax2.invert_yaxis(); ax2.set_title("Feature Importance", color=TEXT_CLR)
                    style_fig(fig2); st.pyplot(fig2)

                quality = "excellent" if r2>0.8 else ("good" if r2>0.6 else "moderate")
                st.markdown(f"<div class='success-box'>The <strong>{algo}</strong> model achieved R² = <strong>{r2:.4f}</strong> ({quality} fit). Predictions are off by ~${rmse:,.0f} on average (RMSE).</div>", unsafe_allow_html=True)

            # ── Classification results ──────────────────────────────────────
            else:
                acc  = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec  = recall_score(y_test, y_pred)
                f1   = f1_score(y_test, y_pred)

                st.markdown("**Performance Metrics**")
                c1,c2,c3,c4 = st.columns(4)
                for col_obj,val,lbl in [
                    (c1,f"{acc*100:.1f}%","Accuracy"),
                    (c2,f"{prec*100:.1f}%","Precision"),
                    (c3,f"{rec*100:.1f}%","Recall"),
                    (c4,f"{f1*100:.1f}%","F1-Score"),
                ]:
                    col_obj.markdown(f"<div class='metric-card'><div class='val'>{val}</div><div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5,4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['Budget','Premium'],
                            yticklabels=['Budget','Premium'],
                            annot_kws={"size":14})
                ax.set_title("Confusion Matrix", color=TEXT_CLR)
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                style_fig(fig); st.pyplot(fig)
                st.markdown(f"<div class='success-box'><strong>{algo}</strong> classified Budget vs Premium cars with {acc*100:.1f}% accuracy. Precision: {prec*100:.1f}%, Recall: {rec*100:.1f}%, F1: {f1*100:.1f}%.</div>", unsafe_allow_html=True)
