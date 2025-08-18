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
- `base_passenger_fare` → Base passenger fare before any tips, tolls, fees, etc. (continuous, **target**)  
- `tips` → Tip amount (continuous)  
- `tolls` → Tolls paid (these are passed to the rider) (continuous)
- `driver_pay` → Driver’s base pay for the trip, not including tips (continuous)
- `cbd_congestion_fee` → New 2025 additional fee imposed on drivers in NYC, passed directly to the rider (continuous)
- `bcf` - Contributions to the black car fund, Uber's fund to pay for driver work benefits, also passed directly to the rider as a 2.5% surcharge of total fare (continuous)
- `congestion_surcharge`  → Surcharge passed to rider (continuous)
- `airport_fee`  → a flat $2.50 fee for pickup or dropoff to airports around NYC
- `shared_request_flag`  → cheaper pricing for riders who accept rideshare, allowing multiple unaffiliated riders to share a ride (categorical)
---

## [Uber Data Analytics Colab Notebook](https://colab.research.google.com/github/annajliu110101/Uber-Driver-Pay-Algorithm/blob/main/notebooks/Exploration.ipynb)  

---

## 📊 Exploratory Data Analysis (EDA)

We performed an initial EDA to better understand our dataset. Below are the three key plots:

### 1. Correlation Matrix
Shows correlations between numerical variables such as trip distance, trip time, fares, tips, and driver pay. This helps identify which variables are strongly related to passenger fares.  
<img width="903" height="803" alt="image" src="https://github.com/user-attachments/assets/e7345391-1699-4cdc-ac89-153f1e188a45" />


---

### 2. Scatter Plot with Linear Regression
Displays the relationship between **trip miles** and **base passenger fare**, with a regression line. We observe a strong positive correlation: longer distances lead to higher fares.  
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/e8be7d24-680e-4c4a-ac75-d4f606cbc0d8" />


---

### 3. Hexbin Heatmap
A density-based visualization of **trip time vs. base passenger fare**, with **driver pay** as the color dimension. This shows clustering of short trips with low fares, while longer trips (30–60 minutes) yield higher fares and driver pay.  
<img width="753" height="590" alt="image" src="https://github.com/user-attachments/assets/a83f8404-87da-4cbe-8603-5ee3f7f3d324" />
)

---

## 🔧 Pre-Processing Data Plan:
- Filter scope of dataset to Uber-only and dropping rows containing Lyft Data.
- Drop platform-specific fields unrelevant to data goals
- Drop rows with essential data missing
- Add derived columns:
    - 'driver_take_home' = driver pay + tips
    - 'final_passenger_fare' = The final charge to the passenger's payment method; including tolls, tips, fees, etc.  Added to better reflect the actual cost of taking an Uber
    - 'trip_profit' = Difference between cost to operate ride (driver's pay) and the payment from passenger.
- Define views of dataframes for analysis.
      - Uber's algorithms are not uniform: airport vs non-airport, rush vs non-peak, distance, dry vs rainy hours, etc.  Averages tend to hide structure which is what we are looking for.  Thus, we create predefined views to isolate more hoomogenous conditions to read cleaner patterns and compare models fairly and be able to explain disparities, variances, etc.  
- Set columns with discrete data as categorical to ensure downstream processing does not read as continuous 

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
