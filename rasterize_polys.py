import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

# === Step 1: Load data ===
polygons = gpd.read_file(r'C:\Users\mfichera\PycharmProjects\aquiferie\combined_reports_bbox_smoothed.shp')
extent = gpd.read_file(r"C:\Users\mfichera\OneDrive - nmt.edu\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData\basemap\NM_dataextent_WGS.shp")

# === Step 2: Reproject to UTM 13N (meters) ===
utm_crs = "EPSG:26913"
polygons = polygons.to_crs(utm_crs)
extent = extent.to_crs(utm_crs)

# === Step 2: Define raster parameters ===
resolution = 100  # meters
bounds = extent.total_bounds  # (minx, miny, maxx, maxy)

# Calculate raster dimensions
width = int((bounds[2] - bounds[0]) / resolution)
height = int((bounds[3] - bounds[1]) / resolution)

# Define transform
transform = from_origin(bounds[0], bounds[3], resolution, resolution)

# === Step 3: Prepare shapes (geometry, value) ===
# Explode multi-polygons into individual polygons if needed
polygons = polygons.explode(index_parts=False)

# Create list of (geometry, value) tuples
shapes = [(geom, value) for geom, value in zip(polygons.geometry, polygons["total"])]

# === Step 4: Rasterize with accumulation ===
# Create an empty array for accumulation
raster = np.zeros((height, width), dtype=np.float32)

# === Step 5: Rasterize and accumulate ===

for geom, value in shapes:
    single_layer = features.rasterize(
        [(geom, value)],
        out_shape=raster.shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.float32
    )
    raster += single_layer

# === Step 6: Save UTM raster temporarily ===
temp_raster_path = r"sum_total_utm.tif"

with rasterio.open(
    temp_raster_path,
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype=raster.dtype,
    crs=polygons.crs,
    transform=transform,
    nodata=0
) as dst:
    dst.write(raster, 1)

print(f"✅ UTM raster saved: {temp_raster_path}")


# === Step 7: Reproject raster to WGS84 ===
final_raster_path = r"C:\Users\mfichera\PycharmProjects\aquiferie\sum_total_wgs84.tif"
dst_crs = "EPSG:4326"

with rasterio.open(temp_raster_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height,
        'nodata': 0
    })

    with rasterio.open(final_raster_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )

print(f"✅ Final WGS84 raster saved: {final_raster_path}")
