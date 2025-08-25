import os
import polars as pl
import pandas as pd
import glob
from pathlib import Path
from pre_processing import process_single_file_lazy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import time

def process_single_file_dataframe(file_path):
    """
    Process a single parquet file and return a polars DataFrame (not lazy).
    """
    print(f"Processing: {Path(file_path).name}")
    lazy_df = process_single_file_lazy(file_path)
    dataframe = lazy_df.collect()  # Convert lazy frame to DataFrame
    print(f"✅ Processed into DataFrame with shape: {dataframe.shape}")
    return dataframe

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)

# Get the base directory and find parquet files
BASE = Path(__file__).parent.resolve()
dataset_paths = glob.glob(f"{BASE}/**/*.parquet", recursive=True)

if dataset_paths:
    print(f"Found {len(dataset_paths)} parquet files")
    print(f"Using first file: {Path(dataset_paths[0]).name}")
    
    # Process single file into DataFrame (not lazy frame)
    single_df = process_single_file_dataframe(dataset_paths[0])
    
else:
    print("No parquet files found!")

single_df = single_df.filter((pl.col('PULocationID').cast(pl.Int32) != 264) & (pl.col('DOLocationID').cast(pl.Int32) != 265))
#single_df = single_df.to_dummies(columns=['PULocationID'])
print(single_df.head(5))
print(single_df.shape)

enhanced_df = single_df.with_columns([
    # Convert datetime to numeric features
    pl.col("request_datetime").dt.hour().alias("request_hour"),
    pl.col("request_datetime").dt.weekday().alias("request_weekday"),
    pl.col("pickup_datetime").dt.hour().alias("pickup_hour"),
    pl.col("pickup_datetime").dt.weekday().alias("pickup_weekday"),
    
    #Wait Time
    (pl.col("pickup_datetime") - pl.col("request_datetime")).dt.total_seconds().alias("wait_time_seconds"),

    #Time Period Indicators (kept exactly as in your code)
    pl.when(pl.col("pickup_datetime").dt.hour().is_between(7, 9)).then(1).otherwise(0).alias("morning_rush"),
    pl.when(pl.col("pickup_datetime").dt.hour().is_between(17, 19)).then(1).otherwise(0).alias("evening_rush"),
    pl.when(pl.col("pickup_datetime").dt.hour().is_between(22, 5)).then(1).otherwise(0).alias("late_night"),
    pl.when(pl.col("pickup_datetime").dt.weekday().is_in([6, 7])).then(1).otherwise(0).alias("weekend"),

    #Distance Categories
    pl.when(pl.col("trip_miles") < 1).then(0)
        .when(pl.col("trip_miles") < 3).then(1)
        .when(pl.col("trip_miles") < 5).then(2)
        .when(pl.col("trip_miles") < 10).then(3)
        .otherwise(4).alias("distance_category"),

    #Duration Categories
    pl.when(pl.col("trip_time") < 300).then(0)  # <5 min
        .when(pl.col("trip_time") < 900).then(1)   # 5-15 min
        .when(pl.col("trip_time") < 1800).then(2)  # 15-30 min
        .otherwise(3).alias("duration_category"),
])

#Choose top 20 pickup and dropoff locations
pickup_counts = single_df.group_by("PULocationID").len().sort("len", descending=True)
dropoff_counts = single_df.group_by("DOLocationID").len().sort("len", descending=True)
    
top_pickup_locations = pickup_counts.limit(20)["PULocationID"].to_list()
top_dropoff_locations = dropoff_counts.limit(20)["DOLocationID"].to_list()

# Create features for top pickup locations
for loc in top_pickup_locations:
    enhanced_df = enhanced_df.with_columns(
        pl.when(pl.col("PULocationID") == loc).then(1).otherwise(0).alias(f"PULocationID_{loc}")
    )
    
# Create features for top dropoff locations
for loc in top_dropoff_locations:
    enhanced_df = enhanced_df.with_columns(
        pl.when(pl.col("DOLocationID") == loc).then(1).otherwise(0).alias(f"DOLocationID_{loc}")
    )

X = enhanced_df.select([
    "trip_time","trip_miles","wait_time_seconds","morning_rush","evening_rush",
    "late_night","weekend","request_hour","request_weekday","pickup_hour","pickup_weekday",
    "distance_category","duration_category", 
] + [f"PULocationID_{loc}" for loc in top_pickup_locations] + [f"DOLocationID_{loc}" for loc in top_dropoff_locations])
y = enhanced_df.select("base_passenger_fare")

print("Shape of X:", X.shape)

# Convert polars DataFrames to pandas for sklearn
X_pandas = X.to_pandas()
y_pandas = y.to_pandas()

# Store feature names for the model
feature_names = X_pandas.columns.tolist()

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X_pandas, y_pandas, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ============================================================================
# GPU-ACCELERATED DECISION TREE REGRESSION IMPLEMENTATION
# ============================================================================

print("\n" + "="*50)
print("DECISION TREE REGRESSION")
print("="*50)

print(f"\nDataset Info:")
print(f"  Training samples: {len(X_train):,}")
print(f"  Validation samples: {len(X_val):,}")
print(f"  Test samples: {len(X_test):,}")
print(f"  Features: {X_train.shape[1]}")

# GPU Training with XGBoost - Hyperparameter Tuning
print(f"\n📊 Decision Tree Model - Hyperparameter Tuning...")

# GPU hyperparameter tuning for max_depth
gpu_max_depths = range(10, 15)  # kept as in your code
gpu_val_errors = []
gpu_training_times = []

print(f"\n🔍 Testing GPU XGBoost with different max_depth values...")

total_gpu_start_time = time.time()

for depth in gpu_max_depths:
    print(f"Training with max_depth={depth}...")
    
    depth_start_time = time.time()
    
    gpu_regressor = xgb.XGBRegressor(
        # Tree parameters
        max_depth=depth,
        n_estimators=150,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        
        # GPU acceleration
        tree_method='hist',
        device='cuda:0',
        
        # Other parameters
        random_state=42
    )
    
    # Train the model
    gpu_regressor.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Make predictions and calculate error
    gpu_val_pred = gpu_regressor.predict(X_val)
    gpu_error = mean_squared_error(y_val, gpu_val_pred)
    gpu_val_errors.append(gpu_error)
    
    depth_training_time = time.time() - depth_start_time
    gpu_training_times.append(depth_training_time)
    
    print(f"  MSE: {gpu_error:.4f}, Time: {depth_training_time:.2f}s")

# Find the best max_depth for GPU
gpu_best_depth = list(gpu_max_depths)[int(np.argmin(gpu_val_errors))]
total_gpu_time = time.time() - total_gpu_start_time

print(f"\n🎯 Best max_depth: {gpu_best_depth}")
print(f"📊 Total hyperparameter tuning time: {total_gpu_time:.2f} seconds")

# ===== Save: Validation Error vs. max_depth =====
depth_list = list(gpu_max_depths)
best_idx = int(np.argmin(gpu_val_errors))

plt.figure(figsize=(8,5))
plt.plot(depth_list, gpu_val_errors, marker='o', label='Validation MSE')
plt.scatter([depth_list[best_idx]], [gpu_val_errors[best_idx]],
            s=120, edgecolor='k', zorder=5, label=f'Best depth = {depth_list[best_idx]}')
plt.xlabel('max_depth')
plt.ylabel('MSE')
plt.title('Validation Error vs. max_depth (XGBoost Regressor)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("figures/val_mse_vs_max_depth.png", dpi=200)
print("Saved figure: figures/val_mse_vs_max_depth.png")

# ===== Shallow vs Best Depth – RMSE Comparison (plot + CSV) =====
from sklearn.metrics import mean_squared_error as _mse

def _fit_rmse_for_depth(depth):
    reg = xgb.XGBRegressor(
        max_depth=depth,
        n_estimators=150,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        device='cuda:0',
        random_state=42
    )
    reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    def rmse(X, y):
        y_pred = reg.predict(X)
        return _mse(y, y_pred, squared=False)

    return rmse(X_train, y_train), rmse(X_val, y_val), rmse(X_test, y_test)

shallow_depth = 2
tr_s, va_s, te_s = _fit_rmse_for_depth(shallow_depth)
tr_b, va_b, te_b = _fit_rmse_for_depth(int(gpu_best_depth))

rmse_df = pd.DataFrame({
    "Model": ["Shallow", "Best"],
    "Depth": [shallow_depth, int(gpu_best_depth)],
    "Train_RMSE": [tr_s, tr_b],
    "Val_RMSE": [va_s, va_b],
    "Test_RMSE": [te_s, te_b]
}).round(3)

print("\n=== Shallow vs Best Depth RMSE ===")
print(rmse_df)

rmse_path = "figures/rmse_shallow_vs_best.csv"
rmse_df.to_csv(rmse_path, index=False)
print(f"Saved table: {rmse_path}")

# Bar plot
labels = ["Train_RMSE", "Val_RMSE", "Test_RMSE"]
x = np.arange(len(labels))
width = 0.36

plt.figure(figsize=(8,5))
plt.bar(x - width/2, [tr_s, va_s, te_s], width, label=f"Shallow (d={shallow_depth})")
plt.bar(x + width/2, [tr_b, va_b, te_b], width, label=f"Best (d={int(gpu_best_depth)})")
plt.xticks(x, labels)
plt.ylabel("RMSE")
plt.title("Shallow vs Best Depth – RMSE Comparison")
plt.grid(True, axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("figures/rmse_shallow_vs_best.png", dpi=200)
print("Saved figure: figures/rmse_shallow_vs_best.png")

# (Your original quick plot/show kept for interactive viewing)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(gpu_max_depths, gpu_val_errors, marker='o', color='green', label='GPU XGBoost')
plt.xlabel('Max Depth')
plt.ylabel('Validation MSE')
plt.title('Validation Error vs Max Depth')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Train final GPU model with best depth
print(f"\n🚀 Training final model with optimal depth ({gpu_best_depth})...")

final_gpu_start_time = time.time()

final_gpu_regressor = xgb.XGBRegressor(
    max_depth=gpu_best_depth,
    n_estimators=100,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    device='cuda:0',  # Specify RTX 3080 (GPU 0)
    random_state=42
)

# Train on training + validation data
X_train_full = pd.concat([X_train, X_val], ignore_index=True)
y_train_full = pd.concat([y_train, y_val], ignore_index=True)

final_gpu_regressor.fit(X_train_full, y_train_full, verbose=False)

# Final predictions and metrics
gpu_test_pred = final_gpu_regressor.predict(X_test)
final_gpu_test_mse = mean_squared_error(y_test, gpu_test_pred)
final_gpu_training_time = time.time() - final_gpu_start_time

print(f"\n🎉 Final Training Results:")
print(f"   Best max_depth: {gpu_best_depth}")
print(f"   Test MSE: {final_gpu_test_mse:.4f}")
print(f"   Test RMSE: ${final_gpu_test_mse**0.5:.2f}")

# Feature importance analysis
print(f"\n📈 Feature Importance (Final Model):")
feature_names = X_pandas.columns.tolist()
importances = final_gpu_regressor.feature_importances_

# Sort features by importance
feature_importance = list(zip(feature_names, importances))
feature_importance.sort(key=lambda x: x[1], reverse=True)

for name, importance in feature_importance:
    print(f"   {name}: {importance:.4f}")

# ===== Feature Importance (Top 20) plot =====
importances_arr = np.array(importances)
idx = np.argsort(importances_arr)[::-1][:20]
top_feats = [feature_names[i] for i in idx]
top_vals = importances_arr[idx]

plt.figure(figsize=(9,7))
plt.barh(range(len(top_feats)), top_vals)
plt.yticks(range(len(top_feats)), top_feats)
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("XGBoost Feature Importance (Top 20)")
plt.tight_layout()
plt.savefig("figures/feature_importance_top20.png", dpi=200)
print("Saved figure: figures/feature_importance_top20.png")

# Performance Summary
print(f"\n📊 PERFORMANCE COMPARISON:")
print(f"   Dataset: {len(X_train_full):,} total samples")
print(f"   Features: {X_train.shape[1]}")
print("-" * 50)

print(f"   XGBoost (Decision Tree):")
print(f"     Best max_depth: {gpu_best_depth}")
print(f"     Test MSE: {final_gpu_test_mse:.4f}")
print(f"     Test RMSE: ${final_gpu_test_mse**0.5:.2f}")
