import os
import pandas as pd
import geopandas as gpd

# === Step 1: Combine CSVs ===
root_dir = "."
combined_df = pd.DataFrame()
target_suffix = "_aquiferinsights_selfeval.csv"
output_csv = "combined_insights.csv"

def get_reports_with_hydrogeo_props():
    # === Step 1: Find and combine matching CSV files ===
    print(f"🔍 Searching for files ending with '{target_suffix}' in '{root_dir}'...")

    csv_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(target_suffix):
                full_path = os.path.join(dirpath, file)
                print(f"📄 Found: {full_path}")
                csv_files.append(full_path)

    if not csv_files:
        print("❌ No matching CSV files found. Check the suffix or directory path.")
        exit()

    # Read and combine all found CSV files
    combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"✅ Combined CSV saved to: {output_csv}")

    # === Step 2: Load and filter shapefile ===
    shapefile_path = "reports_bboxes_oldresearchincluded.shp"
    gdf = gpd.read_file(shapefile_path)

    # Filter where 'Does the_1' == 1
    filtered_gdf = gdf[gdf["Does the_1"] == 1]

    # === Step 3: Join with combined CSV on 'Report' ===
    # Ensure both columns are string type for join
    filtered_gdf["Report"] = filtered_gdf["Report"].astype(str)
    combined_df["Report"] = combined_df["Report"].astype(str)

    # Perform inner join to keep only matching records
    joined_df = pd.merge(filtered_gdf, combined_df, on="Report", how="inner")

    # Show result summary
    print(f"✅ Joined dataframe has {len(joined_df)} rows")

    # Optional: Save the joined result
    joined_df.to_csv("hydrogeoprops_insights.csv", index=False)
    print("✅ Final joined data saved to: hydrogeoprops_insights.csv")

def print_hgprops_answers(file):
    df = pd.read_csv(file)
    question = 'Does the study report specific hydrogeologic properties such as porosity, permeability, storage coefficients, specific storage, specific yield, hydraulic conductivity, and/or transmissivity?'
    for i, (report, value) in enumerate(zip(df['Report'], df[question])):
        print(f'Report: {report}')
        print(f"Row {i}: {value}")


def main():
    print_hgprops_answers(file='hydrogeoprops_insights.csv')

if __name__ == '__main__':
    main()
