# Uber Fare Prediction – Group Project for CSE 151A

## 📄 Abstract
We will be using the **Uber NYC For-Hire Vehicles Trip Data (2021)** dataset, which contains over 11 million trip records per month across New York City, including dispatching base license numbers, pickup and drop-off zones, trip distances, durations, and fare details such as base passenger fare, tips, tolls, and driver pay. Our project applies **decision tree regression** to uncover non-linear relationships between ride features and fare amounts, with the goal of identifying the key attributes that influence pricing. By analyzing temporal, spatial, and trip-specific features, we aim to uncover patterns such as peak-hour surcharges, distance-based pricing, and zone-specific fare differences. The interpretable structure of decision trees will enable us to highlight the most influential factors in fare variation, producing a model that both predicts fares accurately and provides insights into price-optimization strategies for ride-hailing services.

---

## 📌 Dataset
- **Source:** [Uber NYC For-Hire Vehicles Trip Data 2021 – Kaggle](https://www.kaggle.com/datasets/shuhengmo/uber-nyc-forhire-vehicles-trip-data-2021)  
- **Official Documentation:** [NYC TLC Trip Record Data Dictionary](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)  
- **Data Dictionary:** `data_dictionary_trip_records_hvfhs.pdf`  
- **File Example:** `fhvhv_tripdata_2021-01.parquet`  
- **Size:** ~11,908,468 rows × 24 columns (per month)  
- **Format:** Parquet  
- **Scope:** High Volume For-Hire Service trips (Uber, Lyft, Via, etc.) in NYC for 2021  

### Key Features (from TLC dictionary)
- `hvfhs_license_num` → High volume base license number (categorical)  
- `dispatching_base_num` → Dispatching base ID (categorical)  
- `pickup_datetime` → Trip start time (temporal)  
- `dropoff_datetime` → Trip end time (temporal)  
- `PULocationID` → Pickup zone ID (categorical)  
- `DOLocationID` → Drop-off zone ID (categorical)  
- `trip_miles` → Distance in miles (continuous)  
- `trip_time` → Duration in seconds (continuous)  
- `base_passenger_fare` → Base passenger fare (continuous, **target**)  
- `tips` → Tip amount (continuous)  
- `tolls` → Tolls paid (continuous)  
- `driver_pay` → Driver’s pay for the trip (continuous)  

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

Install dependencies via:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
