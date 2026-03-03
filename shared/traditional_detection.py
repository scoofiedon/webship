import numpy as np
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
from scipy.ndimage import uniform_filter, label, binary_dilation
from pathlib import Path
import json

# ============================================================
# 1. LOAD SAR IMAGE
# ============================================================
def load_sar(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        profile = src.profile
    return data, transform, crs, profile

# ============================================================
# 2. BUILD GSHHG LAND MASK (recommended)
# ============================================================
def build_gshhg_land_mask(sar_path, gshhg_shp_path, buffer_deg=0.002):
    """
    Creates a land mask aligned to the SAR image using GSHHG shorelines.
    
    Args:
        sar_path      : path to terrain-corrected GeoTIFF
        gshhg_shp_path: path to GSHHS_f_L1.shp (full resolution)
        buffer_deg    : inland buffer in degrees (~200m at equator)
                        helps mask nearshore clutter and small islands
    Returns:
        land_mask : boolean numpy array (True = land)
    """
    with rasterio.open(sar_path) as src:
        transform = src.transform
        shape = (src.height, src.width)
        bounds = src.bounds
        sar_crs = src.crs

    # Load GSHHG and reproject to match SAR CRS
    gshhs = gpd.read_file(gshhg_shp_path, bbox=(
        bounds.left, bounds.bottom, bounds.right, bounds.top
    ))

    if gshhs.empty:
        print("No land polygons found in scene extent — open ocean image.")
        return np.zeros(shape, dtype=bool)

    gshhs = gshhs.to_crs(sar_crs)

    # Optional: dilate land boundary inland by buffer to catch nearshore clutter
    if buffer_deg > 0:
        gshhs['geometry'] = gshhs.geometry.buffer(buffer_deg)

    # Rasterize: True where land
    land_mask = geometry_mask(
        geometries=gshhs.geometry,
        out_shape=shape,
        transform=transform,
        invert=True   # True = land (inside polygon)
    )

    return land_mask

# ============================================================
# 3. PREPROCESSING (speckle reduction)
# ============================================================
def preprocess(data, look_window=5):
    """Convert from dB if needed, apply multi-look averaging."""
    # Convert dB → linear power if data looks like dB (typically -30 to +5 range)
    if data.mean() < 0:
        data = 10 ** (data / 10)
    multi_looked = uniform_filter(data, size=look_window)
    return multi_looked

# ============================================================
# 4. CA-CFAR DETECTOR (vectorized)
# ============================================================
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

# ============================================================
# 5. APPLY LAND MASK + SAFETY DILATION
# ============================================================
def apply_land_mask(detections, land_mask, extra_dilation_px=3):
    """
    Remove detections on land.
    extra_dilation_px: extra erosion of sea area near coast (removes
    nearshore false alarms even if GSHHG boundary is slightly off).
    """
    if extra_dilation_px > 0:
        dilated_land = binary_dilation(
            land_mask,
            iterations=extra_dilation_px
        )
    else:
        dilated_land = land_mask
    return detections & ~dilated_land

# ============================================================
# 6. BLOB EXTRACTION
# ============================================================
def extract_ships(detections, min_size=4, max_size=500):
    """Label connected components and filter by realistic ship sizes."""
    labeled, n_features = label(detections)
    ships = []
    for i in range(1, n_features + 1):
        blob = np.where(labeled == i)
        size = len(blob[0])
        if min_size <= size <= max_size:
            cy = int(np.mean(blob[0]))
            cx = int(np.mean(blob[1]))
            ships.append({
                'centroid_px': (cy, cx),
                'size_px': size,
            })
    return ships

# ============================================================
# 7. FULL PIPELINE
# ============================================================
def detect_ships(
    sar_path,
    gshhg_shp_path=None,     # None → uses quick land mask fallback
    pfa=1e-6,
    guard=4,
    train=16,
    buffer_deg=0.002,
    extra_dilation_px=3,
):
    raw, transform, crs, _ = load_sar(sar_path)
    img = preprocess(raw)
    land_mask = build_gshhg_land_mask(sar_path, gshhg_shp_path, buffer_deg)
    detections = ca_cfar_fast(img, guard=guard, train=train, pfa=pfa)
    detections = apply_land_mask(detections, land_mask, extra_dilation_px)
    ships = extract_ships(detections)
    return ships, detections, land_mask

# ============================================================
# 8. INTEGRATION WITH EXISTING INFRASTRUCTURE
# ============================================================
def run_traditional_job(job_id: str, image_path: str, params: dict):
    """
    Run traditional detection job with the same interface as neural network jobs.
    Compatible with existing job management system.
    """
    import os
    import time
    import json
    from pathlib import Path
    import rasterio
    from PIL import Image
    import numpy as np

    job_dir = Path(params.get('jobs_dir', '/jobs')) / job_id
    update_status = lambda status, progress=0, msg='': _update_status(job_dir, status, progress, msg)

    try:
        update_status('running', 5, 'Starting traditional detection...')

        # Parameters
        gshhg_path = params.get('gshhg_path')
        pfa = params.get('pfa', 1e-6)
        guard = params.get('guard', 4)
        train = params.get('train', 16)
        buffer_deg = params.get('buffer_deg', 0.002)
        extra_dilation_px = params.get('extra_dilation_px', 3)

        # Run detection
        ships, detections, land_mask = detect_ships(
            image_path,
            gshhg_shp_path=gshhg_path,
            pfa=pfa,
            guard=guard,
            train=train,
            buffer_deg=buffer_deg,
            extra_dilation_px=extra_dilation_px
        )

        # Load image metadata for GeoJSON
        with rasterio.open(image_path) as src:
            transform = src.transform
            crs = src.crs
            H, W = src.height, src.width

        # Convert to GeoJSON format (same as neural network jobs)
        features = []
        for ship in ships:
            cy, cx = ship['centroid_px']
            lon, lat = rasterio.transform.xy(transform, cy, cx)
            
            features.append({
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
                'properties': {
                    'score': 1.0,  # Traditional methods don't produce confidence scores
                    'confidence_bin': '90-100',  # Default high confidence
                    'label': 0,  # Ship class
                    'pixel_x': cx,
                    'pixel_y': cy,
                    'box_width': int(np.sqrt(ship['size_px'])),  # Approximate
                    'box_height': int(np.sqrt(ship['size_px'])),  # Approximate
                    'method': 'ca_cfar'  # Track detection method
                }
            })

        # Save results
        geojson = {
            'type': 'FeatureCollection',
            'crs': {'type': 'name', 'properties': {'name': crs.to_string()}},
            'features': features
        }
        with open(job_dir / 'result.geojson', 'w') as f:
            json.dump(geojson, f, indent=2)

        # Create preview with detections
        update_status('running', 95, 'Generating preview...')
        _create_traditional_preview(job_dir, image_path, ships, detections)

        update_status('done', 100, f'{len(ships)} ships detected (CA-CFAR)')

    except Exception as e:
        import traceback
        update_status('error', 0, str(e))

def _update_status(job_dir: Path, status: str, progress: int = 0, message: str = ''):
    """Helper function to update job status."""
    with open(job_dir / 'status.json', 'w') as f:
        json.dump({
            'status': status,
            'progress': progress,
            'message': message
        }, f)

def _create_traditional_preview(job_dir: Path, image_path: str, ships, detections):
    """Create preview image showing traditional detection results."""
    from PIL import Image, ImageDraw
    import numpy as np
    import rasterio

    # Load and preprocess image for preview
    with rasterio.open(image_path) as src:
        image_np = src.read(1).astype(np.float32)
        H, W = src.height, src.width

    # Convert to uint8 for display
    if image_np.mean() < 0:  # If in dB
        image_np = 10 ** (image_np / 10)
    image_np = np.clip((image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-10) * 255, 0, 255).astype(np.uint8)
    
    # Create RGB image
    preview_img = Image.fromarray(image_np, mode='L').convert('RGB')
    draw = ImageDraw.Draw(preview_img)

    # Draw detections
    for ship in ships:
        cy, cx = ship['centroid_px']
        size = int(np.sqrt(ship['size_px']))
        
        # Draw bounding box
        x1, y1 = cx - size//2, cy - size//2
        x2, y2 = cx + size//2, cy + size//2
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x1, max(0, y1 - 11)), 'Ship', fill=(255, 0, 0))

    # Save preview
    preview_img.save(job_dir / 'preview.png', format='PNG')
    
    # Also create result preview (same format as neural network jobs)
    result_img = preview_img.copy()
    result_img.save(job_dir / 'preview_result.png', format='PNG')