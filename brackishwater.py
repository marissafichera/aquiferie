import os
import pandas as pd
import geopandas as gpd
import re

# Define paths
root_dir = "."  # Change if needed
shapefile_path = "combined_reports_bbox.shp"
output_shapefile = "BrackishWaterRegions_AI.shp"

# Initialize list to collect filtered DataFrames
all_filtered = []

# Recursive search for matching CSV files
for dirpath, _, filenames in os.walk(root_dir):
    for fname in filenames:
        if fname.endswith("_aquiferinsights_set2.csv"):
            match = re.match(r"(.*?)_aquiferinsights_set2\.csv", fname)
            if match:
                basin = match.group(1)
                full_path = os.path.join(dirpath, fname)

                # Load CSV
                df = pd.read_csv(full_path)

                # Trim columns up to "o3-mini Evaluation"
                if "o3-mini Evaluation" in df.columns:
                    col_index = df.columns.get_loc("o3-mini Evaluation")
                    df = df.iloc[:, :col_index]

                # Filter for rows with brackish water = "yes"
                brackish_col = [col for col in df.columns if "brackish" in col.lower()]
                if not brackish_col:
                    continue
                df = df[df[brackish_col[0]].str.strip().str.lower() == "yes"]

                if df.empty:
                    continue

                # Add SourceBasin
                df["SourceBasin"] = basin

                # Add SimpleDepth column
                depth_col = [col for col in df.columns if "are the primary geologic units" in col.lower()]
                if depth_col:
                    def simplify_depth(val):
                        val = str(val).lower()
                        if "both" in val:
                            return "Both"
                        elif "shallow" in val:
                            return "Shallow"
                        elif "deep" in val:
                            return "Deep"
                        elif "not specified" in val:
                            return "Not specified"
                        return "Not specified"

                    df["SimpleDepth"] = df[depth_col[0]].apply(simplify_depth)
                else:
                    df["SimpleDepth"] = "Not specified"

                all_filtered.append(df)

# Combine all filtered data
if all_filtered:
    combined_df = pd.concat(all_filtered, ignore_index=True)

    # Load shapefile and join on 'Report'
    gdf = gpd.read_file(shapefile_path)
    merged = gdf.merge(combined_df, on="Report", how="inner")

    # Save output
    merged.to_file(output_shapefile)
    print(f"✅ Exported to: {output_shapefile}")
else:
    print("⚠️ No rows matched the brackish water criteria.")
