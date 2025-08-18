# Uber Fare Prediction – Milestone 2

## 📄 Abstract
For Milestone 2, we continue working with the **Uber NYC For-Hire Vehicles Trip Data (2025)**.  
Due to file size limitations, we are currently analyzing only **January 2025** trip data as a representative subset, while the full dataset (12 months) will be used later for training and testing.  

Our focus in this milestone is **exploratory data analysis (EDA)** to understand the relationships between trip characteristics and fare amounts. We specifically look at how variables such as distance, time, tips, tolls, and driver pay influence the **base passenger fare**. The analysis helps uncover pricing dynamics like peak-hour surcharges, distance-based pricing, and spatial-temporal patterns.

---

## 📌 Dataset
- **Source:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)  
- **Format:** Parquet  
- **Subset Used for Milestone 2:** `fhvhv_tripdata_2025-01.parquet`  
- **Size:** ~15,356,455 rows × 12 columns (January 2025 only)  
- **Scope:** High Volume For-Hire Service (Uber) trips in NYC  

### Features
- `pickup_datetime` → Trip start time (temporal)  
- `dropoff_datetime` → Trip end time (temporal)  
- `PULocationID` → Pickup zone ID (categorical)  
- `DOLocationID` → Drop-off zone ID (categorical)  
- `trip_miles` → Distance in miles (continuous)  
- `trip_time` → Duration in seconds (continuous)  
- `base_passenger_fare` → Base passenger fare before tips/tolls/fees (**target**)  
- `tips` → Tip amount (continuous)  
- `tolls` → Tolls paid (continuous)  
- `driver_pay` → Driver’s base pay for the trip (continuous)  
- `congestion_surcharge` → Extra NYC congestion fee (continuous)  
- `airport_fee` → Flat $2.50 for airport pickups/drop-offs (continuous)  

---

## 📊 Exploratory Data Analysis (EDA)

We performed an initial EDA to better understand our dataset. Below are the three key plots:

### 1. Correlation Matrix
Shows correlations between numerical variables such as trip distance, trip time, fares, tips, and driver pay. This helps identify which variables are strongly related to passenger fares.  
![Correlation Matrix](figures/correlation_matrix.png)

---

### 2. Scatter Plot with Linear Regression
Displays the relationship between **trip miles** and **base passenger fare**, with a regression line. We observe a strong positive correlation: longer distances lead to higher fares.  
![Trip Distance vs Cost](figures/linear_regression_tripmiles_fare.png)

---

### 3. Hexbin Heatmap
A density-based visualization of **trip time vs. base passenger fare**, with **driver pay** as the color dimension. This shows clustering of short trips with low fares, while longer trips (30–60 minutes) yield higher fares and driver pay.  
![Hexbin Heatmap](figures/hexbin_triptime_fare_driverpay.png)

---

## ⚙️ Environment Setup
To reproduce our work, install the following:

- Python 3.9+  
- Jupyter Notebook  
- Required packages:  
  - `pandas`  
  - `numpy`  
  - `matplotlib`  
  - `seaborn`  
  - `scikit-learn`  
  - `pyarrow`

Install dependencies via:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn pyarrow
