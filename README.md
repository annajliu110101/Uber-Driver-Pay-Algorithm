# Uber Fare Prediction – Milestone 3

## 📄 Abstract
For Milestone 3, we move from EDA to **supervised modeling** on the NYC TLC High-Volume For-Hire Vehicle (Uber) data (2024–2025).  
Our goal is to predict **base passenger fare** using engineered temporal, spatial, and ride features. This milestone delivers:
- completed preprocessing & feature engineering,
- a first model (decision tree–based XGBoost regressor),
- hyperparameter tuning and fitting analysis (under/overfitting),
- evaluation on train/validation/test splits,
- conclusions and next-step plans.

---

## 📌 Dataset
- **Source:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)  
- **Format:** Parquet  
- **Scope Available:** Aug 2024 → Jun 2025 (~163M trips combined)  
- **Subset Used for training in this milestone:** one monthly file (~15M rows)

**Core fields**
- Temporal: `pickup_datetime`, `dropoff_datetime`, `request_datetime`  
- Ride: `trip_miles`, `trip_time`  
- Locations: `PULocationID`, `DOLocationID`  
- **Target:** `base_passenger_fare`

---

## 🔧 Preprocessing & Feature Engineering
1. **Filtering**
   - Keep Uber HV0003 trips (handled upstream in preprocessing pipeline).
   - Remove invalid zones: `PULocationID=264`, `DOLocationID=265`.

2. **Temporal expansions**
   - `request_hour`, `request_weekday`, `pickup_hour`, `pickup_weekday`.

3. **Derived signals**
   - `wait_time_seconds = pickup_datetime – request_datetime`.
   - Rush-hour flags: `morning_rush` (7–9), `evening_rush` (17–19), `late_night` (22–5), `weekend`.

4. **Binning (feature expansion)**
   - `distance_category` from `trip_miles`.
   - `duration_category` from `trip_time`.

5. **Categorical encoding**
   - One-hot of **top 20** pickup zones and **top 20** dropoff zones.

**Final feature count:** **53** (for the training subset).

---

## 🤖 Model 1 — Decision Tree (XGBoost Regressor)
- **Model:** `XGBRegressor` (tree-based gradient boosting)
- **Acceleration:** `tree_method=hist`, `device=cuda:0`
- **Tuning sweep:** `max_depth ∈ [10, 14]` (validated)
- **Fixed params during sweep:** `n_estimators=150`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`

### 📈 Key Results
- **Best depth:** **12**  
- **Validation behavior:** error decreases up to depth ≈ 12, then begins to rise (overfitting).  
- **Test metrics (final model, depth=12):**  
  - **MSE:** **96.62**  
  - **RMSE:** **$9.83**

---

## 🧪 Fitting Analysis (Under/Overfitting)
- **Shallow trees (depth ≤ 5):** **Underfitting** (high bias; higher Val MSE).  
- **Depth = 12:** **Balanced** (lowest Val MSE; best generalization).  
- **Deeper trees (depth ≥ 13):** **Overfitting** (Val MSE begins to increase).

> Second-model comparison: we explicitly compare a **shallow** model (e.g., `max_depth=2`) vs. the **best** (`max_depth=12`) and report Train/Val/Test RMSE to illustrate bias–variance tradeoff. (See table/plot placeholders below; fill after running the code that saves figures.)

---

## 📊 Feature Importance (Final Model)
Top drivers of predicted fare:
1. `distance_category`  
2. `trip_miles`  
3. `DOLocationID_265`  
4. `duration_category`  
5. `trip_time`  

Location one-hots rank highly as well, indicating strong **spatial pricing** effects.

---

## 📷 Visuals (to be added after generating figures)
Add these files to the repo under `figures/` and they’ll render inline.

- **Validation error vs max depth**  
  `![Validation Error vs Depth](figures/val_mse_vs_max_depth.png)`

- **RMSE comparison: shallow vs best depth**  
  `![Shallow vs Best RMSE](figures/rmse_shallow_vs_best.png)`  
  CSV: `figures/rmse_shallow_vs_best.csv`

- **Top-20 feature importances (final model)**  
  `![Top-20 Importances](figures/feature_importance_top20.png)`

*(Optional single summary panel: `figures/milestone3_summary.png`)*

---

## ✅ Conclusion
- Our first decision tree–based model (XGBoost) achieves **Test RMSE ≈ $9.83** on ~1.43M held-out trips.  
- **Distance**, **duration**, and **spatial** features dominate fare prediction.  
- Depth tuning shows the classic U-shape validation curve; **depth=12** offers the best balance.

### Next Models / Improvements
- Extend tuning grid (`n_estimators`, `learning_rate`, regularization).  
- Add richer temporal context (holidays, surge windows, weather).  
- Benchmark additional models per assignment: **KNN**, **SVM (RBF)**, **Naive Bayes** (for binned-fare classification), **Decision Trees/Random Forest** variants.  
- Scale out to all months for robustness; evaluate per-slice (airport vs. city, rush vs. off-peak).

---

## ⚙️ Repro (local or Codespaces)
```bash
# minimal environment
pip install polars pandas scikit-learn xgboost matplotlib pyarrow

# run training (generates figures into ./figures)
python decisiontree.py

# commit visuals
git add figures/*.png figures/*.csv
git commit -m "Add milestone 3 visuals"
git push
