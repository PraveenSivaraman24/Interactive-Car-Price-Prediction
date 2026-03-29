# 🚗 Interactive Car Price Prediction – End-To-End Data Science Playground

**Interactive Car Price Prediction** is a Streamlit-based interactive web application that automates the entire data science workflow for the Car Price Prediction dataset. It simplifies automotive data analysis by combining data understanding, cleaning, filtering, visualization, and machine learning prediction into a single intuitive interface.

---

## 🚀 Features

### 📂 Fixed Dataset Input
- Pre-loaded Car Price Prediction dataset (2,500 Records · 10 Features)
- Instant load — no upload needed

### 🔍 Automatic Data Detection
- Identifies numeric and categorical columns automatically

### 📊 Data Understanding
- Dataset shape (rows & columns)
- Missing values & duplicates
- Data types and descriptive statistics
- Missing value visualization

### 🧹 Data Cleaning
- Removes duplicate rows
- Fills missing values:
  - Numeric → Mean
  - Categorical → Mode
- Detects and removes outliers via the IQR method
- Maintains a detailed cleaning log

### 🎯 Dynamic Data Filtering
- Multi-select filters: Brand, Fuel Type, Transmission, Condition
- Range sliders: Year and Price
- View retained vs removed rows
- Preview filtered dataset

### 📈 Exploratory Data Analysis (EDA)
- Statistical summaries (mean, median, std, min, max)
- Correlation matrix with top relationships
- Categorical value counts

### 📉 Interactive Visualizations
- Heatmap
- Histogram + KDE
- Scatter plot with trendline
- Line chart
- Bar chart
- Box plot
- Auto-generated insights per chart

### 🤖 Machine Learning Modelling

**Regression (Predict Car Price)**
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

**Metrics:** R², RMSE, MAE, Accuracy Proxy, Residual Plot, Feature Importance

**Classification (Budget vs Premium)**
- Logistic Regression (Price > Median → Premium)

**Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## 📂 Dataset

**File:** `car_price_prediction_.csv`
**Records:** 2,500 | **Features:** 10

| Column | Type | Description |
|--------|------|-------------|
| Car ID | int | Unique identifier |
| Brand | categorical | Tesla, BMW, Audi, Ford, Honda, Mercedes, Toyota |
| Year | int | Manufacturing year (2000–2023) |
| Engine Size | float | Engine displacement (litres) |
| Fuel Type | categorical | Petrol, Diesel, Electric, Hybrid |
| Transmission | categorical | Manual, Automatic |
| Mileage | int | Odometer reading (km) |
| Condition | categorical | New, Used, Like New |
| Price | float | Car price in USD |
| Model | categorical | Specific model name |

---

## 🛠 Tech Stack

| Library | Purpose |
|---------|---------|
| `streamlit` | Interactive web UI |
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Charts and plots |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | ML models, preprocessing, evaluation |

---

## 🚀 How to Run

### Step 1 — Install Dependencies
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```
Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 2 — Place the Dataset
Ensure `car_price_prediction_.csv` is in the same directory as `app.py`.

### Step 3 — Launch the App
```bash
streamlit run app.py
```

The app opens automatically at: **http://localhost:8501**

---

## 📁 Project Structure

```
Interactive-Car-Price-Prediction/
├── app.py                      # Main Streamlit application
├── car_price_prediction_.csv   # Dataset (2500 records)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── Readme.txt                  # Quick-start guide for exam/viva
```

---

## 🎨 UI Design

- Dark-themed interface with custom CSS
- `Space Mono` monospace font for headers and metric values
- `DM Sans` clean sans-serif for body text
- Colour palette: `#64b4ff` (blue), `#00c88c` (green), `#ffb432` (amber)
- Sidebar-driven navigation workflow
- Metric cards, insight boxes, and section dividers throughout

---

## 🏫 College

**M.Kumarasamy College of Engineering**
NAAC Accredited Autonomous Institution
Approved by AICTE & Affiliated to Anna University
Thalavapalayam, Karur, Tamilnadu.
