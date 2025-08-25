from matplotlib.font_manager import font_scalings
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.json as js

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import table
from mpl_toolkits.mplot3d import Axes3D

from datetime import datetime, time
from meteostat import Stations, Daily, Hourly, Point
from geopy.geocoders import Nominatim

import polars as pl
import polars.selectors as cs

import glob
# from _principal_componenet_analysis import pca_lazy

BASE = Path(__file__).parent.resolve()

exclude = ['hvfhs_license_num','dispatching_base_num', 'originating_base_num', 'shared_match_flag', 'access_a_ride_flag', 'wav_request_flag', 'wav_match_flag']

weekday_names = {1: "Monday",
                 2: "Tuesday",
                 3: "Wednesday",
                 4: "Thursday",
                 5: "Friday",
                 6: "Saturday",
                 7: "Sunday"}

cat = pl.scan_csv(f'{BASE}/**/*taxi_zone_lookup.csv')
ids = cat.select(pl.col('LocationID').sort().cast(pl.String))
loc_enum = pl.Enum(cat.select(pl.col('LocationID').sort().cast(pl.String)).collect().to_series())

numeric_col = ['trip_miles', 'trip_time', 'base_passenger_fare', 'tolls', 'bcf', 'sales_tax', 'congestion_surcharge', 'airport_fee', 'tips', 'driver_pay', 'cbd_congestion_fee']
datetime_prefix = ['pickup', 'dropoff', 'request']  # 'on_scene' not in data
datetime_col = ['pickup_datetime', 'dropoff_datetime', 'request_datetime']  # 'on_scene_datetime' not in data
boolean_col = ['shared_request_flag']  # Only include columns that actually exist in the data
categories = ['PULocationID', 'DOLocationID']
final_rider_cost_cols = ['base_passenger_fare', 'tolls', 'bcf', 'sales_tax', 'cbd_congestion_fee', 'airport_fee']

def get_taxi_lookup():
    cat = pl.scan_csv(f'{BASE}/**/*taxi_zone_lookup.csv')
    ids = cat.select(pl.col('LocationID').sort().cast(pl.String))
    loc_enum = pl.Enum(cat.select(pl.col('LocationID').sort().cast(pl.String)).collect().to_series())
    return cat, ids, loc_enum

def filter_out_zeros(lazy_df:pl.LazyFrame, strict_col = ['base_passenger_fare']):
    return lazy_df.filter(cs.by_name(strict_col) != 0)

def filter_for_pos(lazy_df:pl.LazyFrame, strict_col = ['base_passenger_fare']):
    return lazy_df.filter(cs.by_name(strict_col) > 0)

def drop_cols(lazy_df:pl.LazyFrame, drop_cols = ['hvfhs_license_num', 'dispatching_base_num', 'originating_base_num']):
    return lazy_df.drop(drop_cols, strict = False)

def filter_rows(lazy_df: pl.LazyFrame, names = ['shared_request_flag', 'shared_match_flag', 'shared_request_flag','access_a_ride_flag', 'wav_request_flag', 'wav_match_flag']):
    return lazy_df.filter(pl.all_horizontal(cs.by_name(*names, require_all = False) == False))

def derive_data(data, schema = None):
    if isinstance(data, str) or isinstance(data, Path):
        data = pl.scan_parquet(data)


    schema = data.collect_schema()
    columns = list(schema.keys())

    if 'hvfhs_license_num' in columns:
        data = data.filter(pl.col('hvfhs_license_num') == 'HV0003').drop(cs.by_name('hvfhs_license_num'))
    if 'cbd_congestion_fee' not in columns:
        data = data.with_columns(pl.zeros(pl.len(), dtype=pl.Float32).alias("cbd_congestion_fee"))
    
    data = data.cast({cs.by_name(*categories): pl.String}).cast({cs.by_dtype(pl.Float64):pl.Float32, cs.by_name(*categories):loc_enum}, strict = True).with_columns(cs.by_name(*boolean_col).replace_strict({"Y": True, "N": False}, return_dtype = pl.Boolean))

    for prefix,c in zip(datetime_prefix, datetime_col):
        data = data.with_columns(cs.by_name(c).dt.date().alias(f"{prefix}_date"), 
                      cs.by_name(c).dt.time().alias(f"{prefix}_time"),
                      cs.by_name(c).dt.weekday().replace_strict(weekday_names, return_dtype = pl.String, default = "0").cast(pl.Categorical).alias(f"{prefix}_day"))

    # Note: wait time calculations disabled because on_scene_datetime is not in the data
    data = data.with_columns(
                              driver_take_home = pl.sum_horizontal(['driver_pay', 'tips']),
                              final_rider_fare = pl.sum_horizontal(*final_rider_cost_cols),
                              wait_time_flag = pl.lit(False),  # Default to False since no on_scene data
                              wait_time_charged = pl.duration(),  # Default to zero duration
                              wait_time_fee = pl.lit(0.0)  # Default to 0

                                    ).with_columns(
                                                   tip_percentage = pl.when(pl.col('tips') == 0).then(0).when(pl.col('final_rider_fare') <= 0).then(pl.col('tips') / pl.sum_horizontal('driver_pay', 'tolls', 'bcf', 'sales_tax', 'cbd_congestion_fee', 'airport_fee')).otherwise(pl.col('tips') / pl.col('final_rider_fare')),

                                                    ).with_columns(net_profit = pl.sum_horizontal('base_passenger_fare', 'wait_time_fee').sub(pl.col('driver_pay')),
                                                                       
                                                                    ).with_columns(uber_cut_percentage = pl.when(pl.col('base_passenger_fare') == 0).then(0).otherwise(pl.col('net_profit')/pl.col('base_passenger_fare')),
                                                                                    final_rider_charge = pl.sum_horizontal(['final_rider_fare', 'tips', 'wait_time_fee']),
                                                                                    )                  
    
    return data

def loading_plot_3d(loadings, feature_names, scale=1.0):
    """
    3D plot of PCA loadings (PC1, PC2, PC3).
    
    loadings : DataFrame or ndarray (n_features x n_components)
        PCA loadings (eigenvectors).
    feature_names : list of str
        Names of features.
    scale : float
        Scale factor for arrow lengths.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # assume loadings has at least 3 PCs
    xs, ys, zs = loadings.iloc[:,0], loadings.iloc[:,1], loadings.iloc[:,2]

    # arrows from origin
    for i, name in enumerate(feature_names):
        ax.quiver(0, 0, 0,
                  xs[i]*scale, ys[i]*scale, zs[i]*scale,
                  arrow_length_ratio=0.1, color="red", alpha=0.7)
        ax.text(xs[i]*scale, ys[i]*scale, zs[i]*scale,
                name, color="black", fontsize=12)
        
    ax.set_xlim(2*xs.min(), 2*xs.max())
    ax.set_ylim(2*ys.min(), 2*ys.max())
    ax.set_zlim(2*zs.min(), 2*zs.max())

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("3D PCA Loadings")
    plt.show()

def loading_plot(loadings, pcx=1, pcy=2, scale=1.0):
    """
    Plot PCA loadings (features only).

    Parameters
    ----------
    loadings : pd.DataFrame
        DataFrame of loadings with features as index,
        PCs as columns (like your .features in skbio).
    pcx, pcy : int
        Which PCs to plot (1-based).
    scale : float
        Scale factor for arrow length (just for display).
    """
    i, j = pcx-1, pcy-1
    plt.figure(figsize=(8, 8))

    for feat, row in loadings.iterrows():
        x, y = row.iloc[i]*scale, row.iloc[j]*scale
        plt.arrow(0, 0, x, y, color="red", alpha=0.7, head_width=0.02)
        plt.text(x*1.1, y*1.1, feat, color="red", fontsize=9)

    plt.axhline(0, color="gray", lw=1)
    plt.axvline(0, color="gray", lw=1)
    plt.xlabel(f"PC{pcx}")
    plt.ylabel(f"PC{pcy}")
    plt.title(f"PCA Loadings Plot (PC{pcx} vs PC{pcy})")
    plt.show()

def run_PCA(path):
    # Temporarily disabled due to missing scikit-bio dependency
    print("PCA functionality temporarily disabled - requires scikit-bio installation")
    # data = pl.scan_parquet(path)
    # data = derive_data(data)
    # results = pca_lazy(data, method = 'eigh', dimensions = 3)
    # print(results.proportion_explained)
    # loading_plot_3d(results.features, results.features.index)

def get_cumulative_data():
    df_i = ['Q3_2024', 'Q4_2024','Q1_2025', 'Q2_2025']
    df_biannual = ['BiA2_2024', 'BiA1_2025']
    df_col = numeric_col + ['final_rider_fare', 'driver_take_home', 'net_profit', 'wait_time_fee']

    data_array = np.zeros(shape = (len(df_col),len(df_i)), dtype = np.float64)
    dataset_paths = glob.glob(f"{BASE}/**/*.parquet", recursive = True)
    dataset_paths.sort(reverse = True)

    percentages = np.zeros(shape = (2,12), dtype = np.float64)
    length = np.zeros((12,2))

    start = 7
    cur = start

    for i, _ in enumerate(df_i):
        while (True):
            if cur == (((start + 3) % 13) or 1):
                start = cur
                break

            path = dataset_paths.pop()
            assert (str(cur) in Path(path).stem.split("-")[1])

            data = derive_data(path)

            percentages[:, cur-1] = data.select(cs.by_name('uber_cut_percentage', 'tip_percentage').sum()).collect().to_numpy()
            length[cur-1, :] = data.select(cs.by_name('uber_cut_percentage', 'tip_percentage').count()).collect().to_numpy()

            data_array[:, i] += data.select(pl.sum(*df_col)).collect().to_numpy().ravel()
            cur = ((cur + 1) % 13) or 1

    percentages /= length.T
    pd.DataFrame(percentages, index = ['Uber Cut Percentage', 'Tip Percentage']).to_csv('percentages.csv')

    pd.DataFrame(data_array, columns = df_i, index = df_col).to_csv('cumulative_data.csv')

def rewrite_dataset():
    dataset_paths = glob.glob(f"{BASE}/**/*.parquet", recursive = True)
    for path in dataset_paths:
        data = pl.scan_parquet(path)
        filtered = data.filter(pl.col('hvfhs_license_num') == 'HV0003')
        filtered.sink_parquet(path)
        
def read_schema():
     dataset_paths = glob.glob(f"{BASE}/**/*.parquet", recursive = True)
     for path in dataset_paths:
        print(path)
        schema = pl.read_parquet_schema(path)
        print(schema)

# Original main function - commented out
# if __name__ == '__main__':
#     dataset_paths = glob.glob(f"{BASE}/**/*.parquet", recursive = True)
#     run_PCA(path = dataset_paths[0])

def process_all_data_lazy():
    """
    Process all parquet files and return a lazy frame with all processed data.
    This keeps the data as a lazy frame for efficient memory usage.
    Handles schema alignment issues across different files.
    """
    dataset_paths = glob.glob(f"{BASE}/**/*.parquet", recursive = True)
    print(f"Found {len(dataset_paths)} parquet files")
    
    # Process each file and collect lazy frames
    lazy_frames = []
    for i, path in enumerate(dataset_paths):
        print(f"Processing file {i+1}/{len(dataset_paths)}: {Path(path).name}")
        try:
            lazy_df = derive_data(path)
            lazy_frames.append(lazy_df)
        except Exception as e:
            print(f"⚠️ Error processing {Path(path).name}: {e}")
            continue
    
    # Concatenate all lazy frames with schema alignment
    if lazy_frames:
        try:
            # Use diagonal concat to handle schema differences
            combined_lazy = pl.concat(lazy_frames, how="diagonal")
            print(f"✅ Combined all {len(lazy_frames)} files into a single lazy frame")
            return combined_lazy
        except Exception as e:
            print(f"❌ Error concatenating files: {e}")
            print("Returning first file only...")
            return lazy_frames[0] if lazy_frames else None
    else:
        print("❌ No files found to process")
        return None

def process_single_file_lazy(file_path):
    """
    Process a single parquet file and return a lazy frame.
    """
    print(f"Processing: {Path(file_path).name}")
    lazy_df = derive_data(file_path)
    print(f"✅ Processed into lazy frame with schema: {list(lazy_df.collect_schema().keys())}")
    return lazy_df

if __name__ == '__main__':
    dataset_paths = glob.glob(f"{BASE}/**/*.parquet", recursive = True)
    print(f"Found {len(dataset_paths)} parquet files:")
    for path in dataset_paths[:3]:  # Show first 3 files
        print(f"  {path}")
    
    if dataset_paths:
        print(f"\n=== Processing Single File (for testing) ===")
        # Process single file as lazy frame
        single_lazy = process_single_file_lazy(dataset_paths[0])
        print(f"Single file lazy frame shape: {single_lazy.select(pl.len()).collect().item()} rows")
        
        print(f"\n=== Processing All Files ===")
        # Process all files as lazy frame
        all_lazy = process_all_data_lazy()
        if all_lazy is not None:
            print(f"✅ Successfully created combined lazy frame!")
            
            first_file_lazy = process_single_file_lazy(dataset_paths[0])
            #print("Average trip distance:", first_file_lazy.select(pl.col('trip_miles').mean()).collect().item())
            #print("Average tip percentage:", first_file_lazy.select(pl.col('tip_percentage').mean()).collect().item())
            #print("Total trips (first file):", first_file_lazy.select(pl.len()).collect().item())
            
            #print(f"   single_lazy.filter(pl.col('trip_miles') > 10)  # Filter long trips")
            #print(f"   single_lazy.group_by('pickup_day').agg(pl.col('tips').mean())  # Average tips by day")
            #print(f"   single_lazy.sink_parquet('processed_data.parquet')  # Save to file")

