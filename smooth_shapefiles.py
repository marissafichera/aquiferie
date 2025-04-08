import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LineString
import numpy as np

# === Function: Chaikin’s corner-cutting algorithm ===
def chaikin_smoothing(geometry, refinements=3):
    if geometry.geom_type == 'Polygon':
        return Polygon(chaikin_coords(list(geometry.exterior.coords), refinements))
    elif geometry.geom_type == 'MultiPolygon':
        return MultiPolygon([Polygon(chaikin_coords(list(p.exterior.coords), refinements)) for p in geometry.geoms])
    else:
        return geometry  # Return unchanged if not polygon

def chaikin_coords(coords, refinements):
    for _ in range(refinements):
        new_coords = []
        for i in range(len(coords) - 1):
            p1 = np.array(coords[i])
            p2 = np.array(coords[i + 1])
            Q = 0.75 * p1 + 0.25 * p2
            R = 0.25 * p1 + 0.75 * p2
            new_coords.extend([tuple(Q), tuple(R)])
        new_coords.append(new_coords[0])  # Close the loop
        coords = new_coords
    return coords

# === Load your combined shapefile ===
gdf = gpd.read_file(r"C:\Users\mfichera\PycharmProjects\aquiferie\combined_reports_bbox.shp")

# === Apply smoothing ===
gdf['geometry'] = gdf['geometry'].apply(lambda geom: chaikin_smoothing(geom, refinements=2))

# === Export smoothed shapefile ===
output_smoothed = r"C:\Users\mfichera\PycharmProjects\aquiferie\combined_reports_bbox_smoothed.shp"
gdf.to_file(output_smoothed)

print(f"✅ Smoothed shapefile saved to: {output_smoothed}")
