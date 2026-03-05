import numpy as np
from rasterio.features import geometry_mask
import geopandas as gpd
from shapely.geometry import box
from scipy.ndimage import uniform_filter
from pyproj.crs import CRS
from pyproj import Transformer

def build_gshhg_land_mask(transform, shape, bounds, sar_crs, gshhg_shp_path, buffer_meters=10.0):
    """
    Creates a land mask aligned to the SAR image using GSHHG shorelines.

    Args:
        transform      : rasterio transform for the SAR image
        shape          : (height, width) of the SAR image
        bounds         : rasterio bounds of the SAR image
        sar_crs        : CRS of the SAR image
        gshhg_shp_path : path to GSHHS_f_L1.shp (full resolution)
        buffer_meters  : inland buffer in meters applied in a UTM projection
                         helps mask nearshore clutter and small islands
    Returns:
        land_mask : boolean numpy array (True = land)
    """
    gshhg_crs = CRS.from_epsg(4326)
    transformer_to_geo = Transformer.from_crs(sar_crs, gshhg_crs, always_xy=True)

    lon_left, lat_bottom = transformer_to_geo.transform(bounds.left, bounds.bottom)
    lon_right, lat_top = transformer_to_geo.transform(bounds.right, bounds.top)

    # Load only polygons intersecting the scene extent
    gshhs = gpd.read_file(gshhg_shp_path, bbox=(
        lon_left, lat_bottom, lon_right, lat_top
    ))

    if gshhs.empty:
        print("No land polygons found in scene extent — open ocean image.")
        return np.zeros(shape, dtype=bool)

    # Buffer in a planar CRS (UTM) where metres are meaningful,
    # then reproject to SAR CRS for rasterization
    if buffer_meters > 0:
        utm_crs = gshhs.to_crs(sar_crs).estimate_utm_crs()
        gshhs = (
            gshhs
            .to_crs(utm_crs)
            .assign(geometry=lambda df: df.geometry.buffer(buffer_meters))
            .to_crs(sar_crs)
        )
    else:
        gshhs = gshhs.to_crs(sar_crs)

    sar_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    gshhs = gshhs.clip(sar_box)
    # Rasterize: True where land
    land_mask = geometry_mask(
        geometries=gshhs.geometry,
        out_shape=shape,
        transform=transform,
        invert=True  # True = land (inside polygon)
    )

    return land_mask
    
def ca_cfar_fast(image, guard=4, train=16, pfa=1e-6):
    """Vectorized Cell-Averaging CFAR."""
    n_outer = (2 * (guard + train) + 1) ** 2
    n_guard = (2 * guard + 1) ** 2
    n_train = n_outer - n_guard

    alpha = n_train * (pfa ** (-1.0 / n_train) - 1)

    sum_outer = uniform_filter(image, size=2*(guard+train)+1) * n_outer
    sum_guard = uniform_filter(image, size=2*guard+1) * n_guard
    noise_estimate = (sum_outer - sum_guard) / n_train

    threshold_map = alpha * noise_estimate
    return image > threshold_map

