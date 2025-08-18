# Uber Fare Prediction – Group Project for CSE 151A

## 📄 Abstract
We will be using the **Uber NYC For-Hire Vehicles Trip Data (2025)** dataset, which contains over 175 million trip records per month across New York City, including dispatching base license numbers, pickup and drop-off zones, trip distances, durations, and fare details such as base passenger fare, tips, tolls, and driver pay. Our project applies **decision tree regression** to uncover non-linear relationships between ride features and fare amounts, with the goal of identifying the key attributes that influence pricing. By analyzing temporal, spatial, and trip-specific features, we aim to uncover patterns such as peak-hour surcharges, distance-based pricing, and zone-specific fare differences. The interpretable structure of decision trees will enable us to highlight the most influential factors in fare variation, producing a model that both predicts fares accurately and provides insights into price-optimization strategies for ride-hailing services.

---

## 📌 Dataset
- **Source:** [Uber NYC For-Hire Vehicles Trip Data 2021 – Kaggle](https://www.kaggle.com/datasets/shuhengmo/uber-nyc-forhire-vehicles-trip-data-2021), [https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page]
- **Official Documentation:** [NYC TLC Trip Record Data Dictionary](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)  
- **Data Dictionary:** `data_dictionary_trip_records_hvfhs.pdf`  
- **File Example:** `fhvhv_tripdata_2021-01.parquet`  
- **Size:** ~11,908,468 rows × 24 columns (per month)  
- **Format:** Parquet  
- **Scope:** High Volume For-Hire Service trips using Uber in NYC for 2025  

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
pip install pandas numpy matplotlib seaborn scikit-learn
```


Pre-Processing Data:
- Filter scope of dataset to Uber-only and dropping rows containing Lyft Data.
- Drop platform-specific fields unrelevant to data goals
- Drop rows with essential data missing
- 
- Add derived columns:
    - 'driver_take_home' = driver pay + tips
    - 'final_passenger_fare' = The final charge to the passenger's payment method; including tolls, tips, fees, etc.  Added to better reflect the actual cost of taking an Uber
    - 'trip_profit' = Difference between cost to operate ride (driver's pay) and the payment from passenger.
- Create views of dataframes to prevent 
- Set columns with discrete data as categorical to ensure downstream processing does not read as continuous 
