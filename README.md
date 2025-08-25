# Uber Fare Prediction – Milestone 3

## 📄 Abstract
For Milestone 3, we extend our work on the **Uber NYC For-Hire Vehicles Trip Data (2024–2025)** by moving from exploratory analysis to **predictive modeling**.  
We focus on building a **GPU-accelerated Decision Tree Regression model (XGBoost)** to predict **base passenger fare** using trip and contextual features.  

This milestone highlights **feature engineering, hyperparameter tuning, and model performance evaluation** on a large-scale dataset.

---

## 📌 Dataset
- **Source:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)  
- **Subset Used for Milestone 3:** `fhvhv_tripdata_2024-07.parquet`  
- **Size:** ~14,328,242 rows × 53 features  
- **Scope:** High Volume For-Hire Service (Uber) trips in NYC  

### Features
Engineered features included:
- Temporal: `request_hour`, `request_weekday`, `pickup_hour`, `pickup_weekday`  
- Derived: `wait_time_seconds`, rush-hour flags, weekend flag  
- Distance & Duration categories  
- Top 20 pickup/dropoff location one-hot features  

**Target Variable:**  
- `base_passenger_fare` (continuous)

---

## ⚙️ Pre-Processing
Steps included:
1. **Filtering** invalid zones (`PULocationID=264`, `DOLocationID=265`).  
2. **Feature engineering** for temporal, categorical, and ride-based patterns.  
3. **One-hot encoding** top pickup and dropoff locations.  
4. Train/validation/test split (80/10/10).  

---

## 🚀 Modeling

We used **XGBoost Regressor with GPU acceleration** (`tree_method='hist', device='cuda:0'`).  
Hyperparameter tuning focused on the **max_depth** parameter, which controls tree complexity.  

### Depth Search Strategy
- Originally tested **depths 1 → 20**.  
- Found best models consistently in **10 → 15** range.  
- To save compute time, narrowed tuning to that range.  
- Final best depth was **12**.  

---

## 📊 Results

### 1. Validation Error vs. Depth
We tested multiple depths, finding that validation error minimized at **depth=12**.  

<img width="1600" height="1000" alt="val_mse_vs_max_depth" src="https://github.com/user-attachments/assets/9a0497f4-6856-4b5a-adc7-51136d24b2cb" />
  
*(Generated from July 2024 dataset, ~14.3M rows)*

---

### 2. Shallow vs. Best Depth Comparison
We compared a shallow tree (`d=2`) vs. the best depth (`d=12`).  
The deeper model showed consistently lower RMSE across training, validation, and testing.  

<img width="780" height="498" alt="image" src="https://github.com/user-attachments/assets/3f58ee4a-e61b-448b-9203-f44f36c08770" />


---

### Final Model Performance (Best depth = 12)
- **Test MSE:** ~96.62  
- **Test RMSE:** ~$9.83  
- **Most important features:**  
  - `distance_category`, `trip_miles`, `trip_time`, duration-related features, and specific pickup/dropoff zones.  

---

## 📈 Key Takeaways
- Distance-based features dominate fare prediction.  
- Temporal features (hour, weekday, rush-hour flags) and pickup/dropoff zones also play a key role.  
- Narrowing the depth range (10–15) significantly reduced tuning time without sacrificing accuracy.  

---

## 🔧 Environment Setup
To reproduce:
```bash
git clone https://github.com/annajliu110101/Uber-Driver-Pay-Algorithm.git
cd Uber-Driver-Pay-Algorithm
pip install -r requirements.txt
