import os
import re
import pandas as pd
import geopandas as gpd

# --- CONFIG ---
root_dir = "./"  # directory containing subfolders with CSVs
# shapefile_path = "reports_bboxes_oldresearchincluded_smoothed.shp"
shapefile_path = 'combined_reports_bbox_smoothed.shp'
output_shapefile = "HydrogeoResearchByDepth.shp"

# --- Utility: strip after 'o3-mini evaluation' from any string ---
def strip_o3(text):
    if isinstance(text, str):
        return text.split("o3-mini evaluation", 1)[0].strip()
    return text

# --- Utility: simplify depth category ---
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

# --- Collect & filter all matching CSVs ---
all_rows = []
for dirpath, _, filenames in os.walk(root_dir):
    for fname in filenames:
        if fname.endswith("_aquiferinsights_set2.csv"):
            basin_match = re.match(r"(.*?)_aquiferinsights_set2\.csv", fname)
            if not basin_match:
                continue

            basin = basin_match.group(1)
            full_path = os.path.join(dirpath, fname)

            try:
                df = pd.read_csv(full_path)

                # Strip o3-mini evaluation content from ALL cells
                df = df.applymap(strip_o3)

                # # Find the brackish water column
                # brackish_col = [c for c in df.columns if "brackish or non-potable water resources" in c.lower()]
                # if not brackish_col:
                #     continue
                #
                # df = df[df[brackish_col[0]].astype(str).str.strip().str.lower().str.startswith("yes")]
                # if df.empty:
                #     continue

                # Add basin column
                df["SourceBasin"] = basin

                # only take old basins
                df = df[(df['SourceBasin'] != 'BrackishWater') | (df['SourceBasin'] != 'AprilReports_Misc')]

                # Extract and simplify depth category
                depth_col = [c for c in df.columns if "primary geologic units" in c.lower()]
                if depth_col:
                    df["SimpleDepth"] = df[depth_col[0]].apply(simplify_depth)
                else:
                    df["SimpleDepth"] = "Not specified"

                all_rows.append(df)

            except Exception as e:
                print(f"⚠️ Error processing {full_path}: {e}")

# --- Combine, merge, and export ---
if all_rows:
    combined_df = pd.concat(all_rows, ignore_index=True)

    # Merge with shapefile
    gdf = gpd.read_file(shapefile_path)
    merged = gdf.merge(combined_df, on="Report", how="inner")

    # Export to shapefile
    merged.to_file(output_shapefile)
    print(f"✅ Exported: {output_shapefile}")
else:
    print("❌ No matching data found.")
