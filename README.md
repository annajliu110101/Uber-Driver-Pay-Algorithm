# Uber Fare Prediction – Milestone 3

## 📄 Abstract
For Milestone 3, we trained our first supervised model on the **Uber NYC For-Hire Vehicles Trip Data (2025)**.  
The focus was to preprocess the dataset, engineer meaningful features, train a baseline model, and evaluate its performance on predicting **base passenger fare**.  

We implemented a **GPU-accelerated Decision Tree Regressor (XGBoost)**, tuning the tree depth (`max_depth`) to study underfitting vs. overfitting.  

---

## 📌 Dataset
- **Source:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)  
- **Subset Used:** `fhvhv_tripdata_2025-01.parquet` (~15.3M rows)  
- **Scope:** Uber trips only, filtered to exclude invalid pickup/dropoff zones.  

### Features Used
- Temporal: `request_hour`, `pickup_hour`, `request_weekday`, `pickup_weekday`  
- Ride context: `wait_time_seconds`, `morning_rush`, `evening_rush`, `late_night`, `weekend`  
- Trip measures: `trip_miles`, `trip_time`, `distance_category`, `duration_category`  
- Spatial: Top 20 most frequent `PULocationID` and `DOLocationID` (one-hot encoded)  

**Target:** `base_passenger_fare`

---

## 🔧 Pre-Processing
1. Filtered Uber-only trips (`hvfhs_license_num = HV0003`) and removed rows with invalid location IDs.  
2. Converted timestamps into numeric and categorical features (hour, weekday, rush-hour indicators).  
3. Engineered categorical distance & duration bins.  
4. One-hot encoded top 20 pickup and dropoff zones.  
5. Final feature set: **53 features**.  

---

## 🚀 Model Training
We trained a **Decision Tree Regressor (XGBoost, GPU-accelerated)**.  

- Training set: ~11.5M samples  
- Validation set: ~1.4M samples  
- Test set: ~1.4M samples  

We originally tested **depths from 1 to 20**, and consistently found that the best-performing models had `max_depth` values between **10 and 15**.  
To save training time and GPU resources, we narrowed the search space in later runs to this range.  

---

## 📊 Results

### Validation Error vs. max_depth
The plot below shows validation MSE across different tree depths.  
Depth = **12** achieved the best tradeoff between underfitting and overfitting.  

<img width="1600" height="1000" alt="val_mse_vs_max_depth" src="https://github.com/user-attachments/assets/edd7d2a5-88bb-44d2-818c-625c2bf3a5a6" />


---

### Final Model Performance
- **Best depth:** 12  
- **Test MSE:** 96.62  
- **Test RMSE:** $9.83  

### Top Feature Importances
1. `distance_category`  
2. `trip_miles`  
3. `DOLocationID_265`  
4. `duration_category`  
5. `trip_time`  

Interpretation: Fare is most strongly driven by **distance-based features** and certain **pickup/dropoff hotspots**.  

---

## 📈 Model Fit Analysis
- Shallow trees (`max_depth < 5`) → **underfit**, with high bias and poor predictive accuracy.  
- Deep trees (`max_depth > 13`) → **overfit**, with validation error increasing.  
- Optimal depth = **12**, balancing training vs. validation performance.  

---

## 📝 Conclusion
Our first supervised model demonstrates that **distance and time are the primary drivers of Uber fares**, but location-based effects also play a strong role.  
The baseline RMSE of **$9.83** is a reasonable starting point given the dataset’s variability.  

### Next Steps
- Experiment with **Random Forests** or **Gradient Boosted Trees**.  
- Add external features (e.g., **weather conditions, surge pricing flags**) for richer context.  
- Perform hyperparameter tuning across learning rate, subsampling, and number of estimators.  

---

## 📂 Repo & Notebooks
- [Preprocessing Pipeline](./pre_processing.py)  
- [Decision Tree Training](./decisiontree.py)  
- [Colab Notebook (Milestone 3)](https://colab.research.google.com/github/annajliu110101/Uber-Driver-Pay-Algorithm/blob/Milestone3/notebooks/DecisionTree.ipynb)  

---
