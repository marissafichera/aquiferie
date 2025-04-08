import geopandas as gpd
import pandas as pd
import glob
import os

# === Step 1: Define paths ===
main_folder = r"C:\Users\mfichera\PycharmProjects\aquiferie"
output_shapefile = r"C:\Users\mfichera\PycharmProjects\aquiferie\combined_reports_bbox.shp"

# === Step 2: Find all shapefiles recursively ===
shapefile_list = glob.glob(os.path.join(main_folder, "*", "*_reports_bbox.shp"))

print(f"Found {len(shapefile_list)} shapefiles.")

# === Step 3: Process each shapefile ===
processed_gdfs = []

for shp_path in shapefile_list:
    # Extract basin name from folder name
    basin_name = os.path.basename(os.path.dirname(shp_path))

    # Read shapefile
    gdf = gpd.read_file(shp_path)

    # Drop unwanted columns if they exist
    columns_to_drop = ["West", "East", "South", "North"]
    gdf = gdf.drop(columns=[col for col in columns_to_drop if col in gdf.columns])

    # Add source basin name as new column
    gdf["SourceBasin"] = basin_name

    processed_gdfs.append(gdf)

# === Step 4: Combine all GeoDataFrames ===
combined_gdf = gpd.GeoDataFrame(pd.concat(processed_gdfs, ignore_index=True), crs=processed_gdfs[0].crs)

# === Step 5: Export combined shapefile ===
combined_gdf.to_file(output_shapefile)

print(f"✅ Combined shapefile exported to: {output_shapefile}")
