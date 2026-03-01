import os, gc, json, shutil
import numpy as np
import torch
import rasterio
from pathlib import Path
from torchvision.ops import nms
from ultralytics import YOLO

DATA_DIR   = Path(os.environ.get('DATA_DIR', '/data'))
MODELS_DIR = Path(os.environ.get('MODELS_DIR', '/models'))

STATS = {
    'VV': {'mean': -12.59, 'std': 5.26},
    'VH': {'mean': -20.26, 'std': 5.91},
    'HH': {'mean': -12.59, 'std': 5.26},
    'HV': {'mean': -20.26, 'std': 5.91},
}

def enhance_contrast(patch: np.ndarray, method: str = 'percentile',
                     percentile_low: float = 2.0, percentile_high: float = 98.0,
                     gamma: float = 1.0, clahe: bool = False) -> np.ndarray:
    """
    Contrast enhancement for SAR imagery.
    method:           'percentile' or 'minmax'
    percentile_low/high: clip range (2-98 is standard for SAR)
    gamma:            <1 brightens, >1 darkens
    clahe:            apply CLAHE after stretch (best for ship visibility)
    """
    arr = patch.astype(np.float32)

    if method == 'percentile':
        lo = np.percentile(arr, percentile_low)
        hi = np.percentile(arr, percentile_high)
    else:
        lo, hi = arr.min(), arr.max()

    arr = np.clip((arr - lo) / (hi - lo + 1e-10), 0, 1)

    # Gamma correction
    if gamma != 1.0:
        arr = np.power(arr, gamma)

    arr = (arr * 255).astype(np.uint8)

    # CLAHE — very effective for SAR ship detection
    if clahe:
        import cv2
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr = clahe_obj.apply(arr)

    return arr


def preprocess_patch(patch: np.ndarray, polarization: str,
                     input_format: str, contrast_params: dict) -> np.ndarray:
    """Preprocess SAR patch → uint8 HWC for YOLO or normalized tensor."""
    pol = polarization.upper()

    if input_format == 'byte':
        arr = patch.astype(np.float32)
    elif input_format == 'linear':
        arr = np.log1p(patch.astype(np.float32))
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10) * 255.0
    elif input_format == 'db':
        arr = patch.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10) * 255.0

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Apply contrast enhancement
    if contrast_params.get('enabled', False):
        arr = enhance_contrast(
            arr,
            method          = contrast_params.get('method', 'percentile'),
            percentile_low  = contrast_params.get('percentile_low', 2.0),
            percentile_high = contrast_params.get('percentile_high', 98.0),
            gamma           = contrast_params.get('gamma', 1.0),
            clahe           = contrast_params.get('clahe', False),
        )

    return np.stack([arr, arr, arr], axis=-1)  # HWC uint8


def score_to_confidence_bin(score):
    pct = int(score * 100)
    if pct < 10: return '<10'
    if pct >= 100: return '90-100'
    low = (pct // 10) * 10
    return f'{low}-{low+10}'


def run_inference_job(job_id: str, image_path: str, model_type: str,
                      params: dict) -> dict:
    """
    Main inference job — runs in worker container.
    Returns path to output GeoJSON.
    """
    job_dir     = DATA_DIR / job_id
    result_path = job_dir / 'result.geojson'
    status_path = job_dir / 'status.json'

    def update_status(status, progress=0, message=''):
        with open(status_path, 'w') as f:
            json.dump({'status': status, 'progress': progress,
                       'message': message}, f)

    try:
        update_status('running', 0, 'Loading model...')

        # Load appropriate model
        if model_type == 'yolo':
            model = YOLO(str(MODELS_DIR / 'yolo26s_best.pt'))
            result = _run_yolo(model, image_path, params, update_status)
        elif model_type == 'fasterrcnn':
            model = torch.jit.load(str(MODELS_DIR / 'fasterrcnn.torchscript'))
            model.eval()
            result = _run_fasterrcnn(model, image_path, params, update_status)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Write GeoJSON
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        update_status('done', 100, f"{len(result['features'])} ships detected")

        # Cleanup model from memory
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return {'status': 'done', 'result_path': str(result_path)}

    except Exception as e:
        update_status('error', 0, str(e))
        raise


def _sliding_window(H, W, window_size, overlap_pct):
    stride   = int(window_size * (1 - overlap_pct))
    y_starts = list(range(0, H - window_size + 1, stride))
    x_starts = list(range(0, W - window_size + 1, stride))
    if not y_starts or y_starts[-1] + window_size < H:
        y_starts.append(max(0, H - window_size))
    if not x_starts or x_starts[-1] + window_size < W:
        x_starts.append(max(0, W - window_size))
    return y_starts, x_starts


def _run_yolo(model, image_path, params, update_status):
    window_size     = params['window_size']
    overlap_pct     = params['overlap_pct']
    score_thresh    = params['score_thresh']
    nms_iou         = params['nms_iou']
    polarization    = params['polarization']
    input_format    = params['input_format']
    contrast_params = params.get('contrast', {})
    device          = params.get('device', 'cpu')

    with rasterio.open(image_path) as src:
        transform = src.transform
        crs       = src.crs
        H, W      = src.height, src.width

    y_starts, x_starts = _sliding_window(H, W, window_size, overlap_pct)
    total = len(y_starts) * len(x_starts)

    all_boxes, all_scores, all_labels = [], [], []

    with rasterio.open(image_path) as src:
        for yi, y in enumerate(y_starts):
            for xi, x in enumerate(x_starts):
                progress = int((yi * len(x_starts) + xi) / total * 90)
                update_status('running', progress, f'Patch {yi*len(x_starts)+xi+1}/{total}')

                window    = rasterio.windows.Window(x, y, window_size, window_size)
                patch     = src.read(1, window=window).astype(np.float32)
                patch_rgb = preprocess_patch(patch, polarization,
                                             input_format, contrast_params)
                del patch

                results = model.predict(
                    patch_rgb.copy(),
                    conf    = score_thresh,
                    iou     = nms_iou,
                    imgsz   = window_size,
                    device  = device,
                    verbose = False,
                )
                del patch_rgb

                result = results[0]
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                boxes  = result.boxes.xyxy.cpu().clone()
                scores = result.boxes.conf.cpu().clone()
                labels = result.boxes.cls.cpu().int().clone()

                boxes[:, [0, 2]] += x
                boxes[:, [1, 3]] += y

                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

    if not all_boxes:
        return {'type': 'FeatureCollection', 'features': []}

    all_boxes  = torch.cat(all_boxes)
    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    keep       = nms(all_boxes.float(), all_scores.float(), nms_iou)
    all_boxes  = all_boxes[keep]
    all_scores = all_scores[keep]
    all_labels = all_labels[keep]

    return _build_geojson(all_boxes, all_scores, all_labels,
                          transform, crs, model.names)


def _build_geojson(boxes, scores, labels, transform, crs, class_names):
    features = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].tolist()
        score  = scores[i].item()
        label  = int(labels[i].item())
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        lon, lat = rasterio.transform.xy(transform, cy, cx)
        features.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
            'properties': {
                'score':          round(score, 4),
                'confidence_bin': score_to_confidence_bin(score),
                'label':          label,
                'class_name':     class_names[label] if class_names else str(label),
                'pixel_x':        round(cx),
                'pixel_y':        round(cy),
                'box_width':      round(x2 - x1),
                'box_height':     round(y2 - y1),
            }
        })
    return {
        'type': 'FeatureCollection',
        'crs':  {'type': 'name', 'properties': {'name': crs.to_string()}},
        'features': features
    }