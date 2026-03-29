╔══════════════════════════════════════════════════════════════╗
║     INTERACTIVE CAR PRICE PREDICTION — Streamlit App        ║
║     Project for Exam / Viva Submission                      ║
╚══════════════════════════════════════════════════════════════╝

📁 FILES IN THIS FOLDER:
   ├── app.py                      ← Main Streamlit app
   ├── car_price_prediction_.csv   ← Dataset (2500 Records · 10 Features)
   ├── requirements.txt            ← Python dependencies
   ├── README.md                   ← Detailed project documentation
   └── Readme.txt                  ← This file

─────────────────────────────────────────────────────────────
🚀 HOW TO RUN (Step-by-Step)
─────────────────────────────────────────────────────────────

STEP 1 — Install Python (if not installed)
   Download from: https://www.python.org/downloads/

STEP 2 — Install required libraries
   Open terminal / command prompt and run:

   pip install streamlit pandas numpy matplotlib seaborn scikit-learn

   OR use the requirements file:
   pip install -r requirements.txt

STEP 3 — Make sure the dataset is in the same folder as app.py
   File: car_price_prediction_.csv

STEP 4 — Run the app
   streamlit run app.py

STEP 5 — App opens in browser automatically at:
   http://localhost:8501

─────────────────────────────────────────────────────────────
📋 WHAT THE APP DOES (5 Modules)
─────────────────────────────────────────────────────────────

Module 1 → Dataset Overview
           Shape, data types, missing values, duplicates,
           descriptive statistics, missing-value visual map

Module 2 → Data Cleaning
           Remove duplicates, fill missing values (mean/mode),
           IQR-based outlier removal, detailed cleaning log

Module 3 → Dynamic Filtering
           Multi-select: Brand, Fuel Type, Transmission, Condition
           Range sliders: Year and Price
           Live row-retained / row-removed metrics

Module 4 → EDA & Visualisations
           Correlation Heatmap, Histogram + KDE, Scatter Plot
           with trendline, Bar Chart, Box Plot, Line Chart
           — each with auto-generated insights

Module 5 → ML Modelling
           Regression → Linear, Ridge, Random Forest,
                         Gradient Boosting (predict Price)
           Classification → Logistic Regression
                             (Budget vs Premium)
           Full evaluation metrics + Residual Plot +
           Feature Importance

─────────────────────────────────────────────────────────────
🎤 VIVA ONE-LINER
─────────────────────────────────────────────────────────────

"My project is an Interactive Car Price Prediction system
built using Python and Streamlit. It takes the Car Price
Prediction dataset through a complete end-to-end pipeline —
data understanding, cleaning, filtering, visualization, and
machine learning — to predict and classify car prices."

─────────────────────────────────────────────────────────────
📦 LIBRARIES USED
─────────────────────────────────────────────────────────────

Library         Purpose
─────────────── ───────────────────────────────────────────
streamlit       Interactive web UI
pandas          Data loading & manipulation
numpy           Numerical operations
matplotlib      Charts and plots
seaborn         Statistical visualizations
scikit-learn    ML models, preprocessing, evaluation

─────────────────────────────────────────────────────────────
📊 DATASET DETAILS
─────────────────────────────────────────────────────────────

File    : car_price_prediction_.csv
Records : 2,500
Features: 10

Columns : Car ID, Brand, Year, Engine Size, Fuel Type,
          Transmission, Mileage, Condition, Price, Model

─────────────────────────────────────────────────────────────
🏫 COLLEGE
─────────────────────────────────────────────────────────────

M.Kumarasamy College of Engineering
NAAC Accredited Autonomous Institution
Approved by AICTE & Affiliated to Anna University
Thalavapalayam, Karur, Tamilnadu.
