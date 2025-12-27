import pandas as pd
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "final_dataset.csv")

# Read the CSV file
df = pd.read_csv(csv_path)

# Define required columns
required_columns = [
    "temp_mean",
    "temp_range",
    "humidity_min",
    "wind_speed_max",
    "pressure_mean",
    "solar_radiation_mean",
    "cloud_cover_mean",
    "occured"
]

# Keep only required columns
df_filtered = df[required_columns]

# Convert occured column to int
df_filtered["occured"] = df_filtered["occured"].astype(int)

# Save the filtered dataset
df_filtered.to_csv(csv_path, index=False)

print("Columns removed successfully!")
print(f"New shape: {df_filtered.shape}")
print(f"Columns: {list(df_filtered.columns)}")
