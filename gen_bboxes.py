import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

# Load the DataFrame from a CSV file
csv_path = r"C:\Users\mfichera\PycharmProjects\aquiferie\EstanciaBasin\bbox.csv"  # <-- change to your actual file path
df = pd.read_csv(csv_path)

# Function to create a rectangular polygon from bounds
def create_rectangle(row):
    return Polygon([
        (row['West'], row['South']),
        (row['West'], row['North']),
        (row['East'], row['North']),
        (row['East'], row['South']),
        (row['West'], row['South'])  # Close the loop
    ])

# Create GeoDataFrame with rectangles
gdf = gpd.GeoDataFrame(
    df[['Report', 'West', 'East', 'South', 'North']],
    geometry=df.apply(create_rectangle, axis=1),
    crs="EPSG:4326"
)

# Save to shapefile
output_path = "bounding_boxes.shp"  # <-- change if needed
gdf.to_file(output_path)

print(f"Shapefile created: {output_path}")
