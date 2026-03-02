import os, gc, json, time
import numpy as np
from pathlib import Path

JOBS_DIR   = Path(os.environ.get('JOBS_DIR',   '/jobs'))
MODELS_DIR = Path(os.environ.get('MODELS_DIR', '/models'))

STATS = {
    'VV': {'mean': -12.59, 'std': 5.26},
    'VH': {'mean': -20.26, 'std': 5.91},
    'HH': {'mean': -12.59, 'std': 5.26},
    'HV': {'mean': -20.26, 'std': 5.91},
}

BIN_COLORS = {
    '<10':    (74,  74,  74),
    '10-20':  (26,  58,  107),
    '20-30':  (26,  82,  118),
    '30-40':  (17,  122, 101),
    '40-50':  (30,  132, 73),
    '50-60':  (212, 172, 13),
    '60-70':  (202, 111, 30),
    '70-80':  (203, 67,  53),
    '80-90':  (231, 76,  60),
    '90-100': (255, 0,   0),
}


def update_status(job_dir: Path, status: str,
                  progress: int = 0, message: str = ''):
    with open(job_dir / 'status.json', 'w') as f:
        json.dump({
            'status':   status,
            'progress': progress,
            'message':  message
        }, f)


def score_to_bin(score: float) -> str:
    pct = int(score * 100)
    if pct < 10:  return '<10'
    if pct >= 100: return '90-100'
    low = (pct // 10) * 10
    return f'{low}-{low+10}'


def enhance_contrast(arr: np.ndarray, method: str = 'percentile',
                     percentile_low: float = 2.0,
                     percentile_high: float = 98.0,
                     gamma: float = 1.0,
                     clahe: bool = True) -> np.ndarray:
    arr = arr.astype(np.float32)
    if method == 'percentile':
        lo = np.percentile(arr, percentile_low)
        hi = np.percentile(arr, percentile_high)
    else:
        lo, hi = arr.min(), arr.max()
    arr = np.clip((arr - lo) / (hi - lo + 1e-10), 0, 1)
    if gamma != 1.0:
        arr = np.power(arr, gamma)
    arr = (arr * 255).astype(np.uint8)
    if clahe:
        import cv2
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr = clahe_obj.apply(arr)
    return arr


def preprocess_patch(patch: np.ndarray, polarization: str,
                     input_format: str,
                     contrast_params: dict,
                     target_size: int = None) -> np.ndarray:
    pol = polarization.upper()
    if input_format == 'byte':
        arr = patch.astype(np.float32)
    elif input_format == 'linear':
        arr = 10.0 * np.log10(np.clip(patch.astype(np.float32), 1e-10, None))
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10) * 255.0
    elif input_format == 'db':
        arr = patch.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10) * 255.0
    else:
        raise ValueError(f'Unknown format: {input_format}')

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Interpolate to target size if specified and patch is smaller
    if target_size is not None and patch.shape[0] < target_size:
        import cv2
        # Interpolate using bicubic interpolation for better quality
        arr = cv2.resize(arr, (target_size, target_size), 
                        interpolation=cv2.INTER_CUBIC)

    if contrast_params.get('enabled', True):
        arr = enhance_contrast(
            arr,
            method          = contrast_params.get('method', 'percentile'),
            percentile_low  = contrast_params.get('percentile_low', 2.0),
            percentile_high = contrast_params.get('percentile_high', 98.0),
            gamma           = contrast_params.get('gamma', 1.0),
            clahe           = contrast_params.get('clahe', True),
        )
    return np.stack([arr, arr, arr], axis=-1)  # HWC uint8


def make_preview(image_np: np.ndarray,
                 contrast_params: dict,
                 max_size: int = 1920) -> 'PIL.Image.Image':
    """Render full scene to preview PNG with contrast enhancement."""
    from PIL import Image
    H, W   = image_np.shape
    scale  = min(max_size / W, max_size / H, 1.0)
    out_w  = max(1, int(W * scale))
    out_h  = max(1, int(H * scale))

    # Downsample
    small  = Image.fromarray(image_np).resize((out_w, out_h),
                                               Image.Resampling.LANCZOS)
    arr    = np.array(small).astype(np.float32)

    # Contrast enhancement
    if contrast_params.get('enabled', True):
        arr = enhance_contrast(
            arr,
            method          = contrast_params.get('method', 'percentile'),
            percentile_low  = contrast_params.get('percentile_low', 2.0),
            percentile_high = contrast_params.get('percentile_high', 98.0),
            gamma           = contrast_params.get('gamma', 1.0),
            clahe           = contrast_params.get('clahe', True),
        )
    else:
        lo  = np.percentile(arr, 2)
        hi  = np.percentile(arr, 98)
        arr = np.clip((arr - lo) / (hi - lo + 1e-10) * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode='L'), scale


def bake_detections(preview_img: 'PIL.Image.Image',
                    features: list,
                    scale: float) -> 'PIL.Image.Image':
    """Draw detections onto preview image and return RGB PIL image."""
    from PIL import Image, ImageDraw
    img  = preview_img.convert('RGB')
    draw = ImageDraw.Draw(img)

    for feat in features:
        props = feat['properties']
        px    = props['pixel_x']
        py    = props['pixel_y']
        bw    = props.get('box_width',  10)
        bh    = props.get('box_height', 10)
        cbin  = props['confidence_bin']
        score = props['score']
        color = BIN_COLORS.get(cbin, (255, 0, 0))

        sx    = int(px * scale)
        sy    = int(py * scale)
        sw    = max(4, int(bw * scale))
        sh    = max(4, int(bh * scale))

        x1, y1 = sx - sw // 2, sy - sh // 2
        x2, y2 = sx + sw // 2, sy + sh // 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(0, y1 - 11)),
                  f'{int(score*100)}%', fill=color)

    # Legend
    lx = img.width  - 85
    ly = img.height - len(BIN_COLORS) * 14 - 10
    draw.rectangle([lx - 4, ly - 4, img.width - 2, img.height - 2],
                   fill=(20, 20, 20))
    for i, (label, color) in enumerate(BIN_COLORS.items()):
        ry = ly + i * 14
        draw.rectangle([lx, ry, lx + 10, ry + 10], fill=color)
        draw.text((lx + 14, ry), label, fill=(220, 220, 220))

    return img


def run_inference_job(job_id: str, image_path: str,
                      model_type: str, params: dict):
    import torch
    import rasterio
    from PIL import Image
    import io

    job_dir = JOBS_DIR / job_id
    update_status(job_dir, 'running', 5, 'Loading image...')

    try:
        # 1. Load full image
        with rasterio.open(image_path) as src:
            image_np  = src.read(1).astype(np.float32)
            transform = src.transform
            crs       = src.crs
            H, W      = src.height, src.width

        contrast_params = params.get('contrast', {
            'enabled': True, 'method': 'percentile',
            'percentile_low': 2.0, 'percentile_high': 98.0,
            'gamma': 1.0, 'clahe': True
        })

        # 2. Make and save preview (FullHD max)
        update_status(job_dir, 'running', 10, 'Generating preview...')
        preview_img, scale = make_preview(image_np, contrast_params)
        preview_path       = job_dir / 'preview.png'
        preview_img.save(preview_path, format='PNG')

        # 3. Delete original image to save space
        import os
        os.remove(image_path)
        update_status(job_dir, 'running', 15, 'Running inference...')

        # 4. Run inference on full image_np (sliding window)
        window_size  = params['window_size']
        overlap_pct  = params['overlap_pct']
        score_thresh = params['score_thresh']
        nms_iou      = params['nms_iou']
        polarization = params['polarization']
        input_format = params['input_format']
        device       = params.get('device', 'cpu')
        
        # Set target size for interpolation - use 800x800 as default for HRSID training
        target_size = params.get('target_size', 800)
        
        # If current window size is smaller than target, use interpolation
        if window_size < target_size:
            print(f"Interpolating patches from {window_size}x{window_size} to {target_size}x{target_size} to match HRSID training data")

        stride   = int(window_size * (1 - overlap_pct))
        y_starts = list(range(0, H - window_size + 1, stride))
        x_starts = list(range(0, W - window_size + 1, stride))
        if not y_starts or y_starts[-1] + window_size < H:
            y_starts.append(max(0, H - window_size))
        if not x_starts or x_starts[-1] + window_size < W:
            x_starts.append(max(0, W - window_size))

        total = len(y_starts) * len(x_starts)

        # Load model
        if model_type == 'yolo':
            from ultralytics import YOLO
            model = YOLO(str(MODELS_DIR / 'best.pt'))
        else:
            from model import get_faster_rcnn
            model = get_faster_rcnn(num_classes=2, pretrained=False, freeze_backbone=False)
            model.load_state_dict(torch.load(str(MODELS_DIR / 'fasterrcnn_weights.pth'), map_location='cpu'))
            model.eval()

        all_boxes, all_scores, all_labels = [], [], []

        for yi, y in enumerate(y_starts):
            for xi, x in enumerate(x_starts):
                patch_num = yi * len(x_starts) + xi + 1
                progress  = 15 + int(patch_num / total * 75)
                update_status(job_dir, 'running', progress,
                              f'Patch {patch_num}/{total}')

                patch     = image_np[y:y+window_size, x:x+window_size]
                patch_rgb = preprocess_patch(patch, polarization,
                                             input_format, contrast_params,
                                             target_size if window_size < target_size else None)
                del patch

                if model_type == 'yolo':
                    results = model.predict(
                        patch_rgb.copy(),
                        conf=score_thresh, iou=nms_iou,
                        imgsz=target_size, device=device, verbose=False,
                    )
                    result = results[0]
                    if result.boxes is None or len(result.boxes) == 0:
                        del patch_rgb
                        continue
                    boxes  = result.boxes.xyxy.cpu().clone()
                    scores = result.boxes.conf.cpu().clone()
                    labels = result.boxes.cls.cpu().int().clone()
                else:
                    import torch
                    tensor = torch.from_numpy(
                        patch_rgb[:,:,0].astype(np.float32) / 255.0
                    ).unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        out = model(tensor)[0]
                    boxes  = out['boxes'].cpu()
                    scores = out['scores'].cpu()
                    labels = out['labels'].cpu().int()
                    keep   = scores > score_thresh
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

                del patch_rgb

                # Adjust coordinates for interpolation if needed
                if window_size < target_size:
                    scale_factor = window_size / target_size
                    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_factor
                    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_factor
                
                boxes[:, [0, 2]] += x
                boxes[:, [1, 3]] += y
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

        del image_np
        gc.collect()

        # 5. Global NMS
        update_status(job_dir, 'running', 92, 'Applying NMS...')
        if all_boxes:
            import torch
            from torchvision.ops import nms as tv_nms
            all_boxes  = torch.cat(all_boxes)
            all_scores = torch.cat(all_scores)
            all_labels = torch.cat(all_labels)
            keep       = tv_nms(all_boxes.float(), all_scores.float(), nms_iou)
            all_boxes  = all_boxes[keep]
            all_scores = all_scores[keep]
            all_labels = all_labels[keep]
            n_det      = len(all_boxes)
        else:
            n_det = 0

        # 6. Build GeoJSON
        update_status(job_dir, 'running', 95, 'Building results...')
        features = []
        if n_det > 0:
            for i in range(n_det):
                x1, y1, x2, y2 = all_boxes[i].tolist()
                score  = all_scores[i].item()
                label  = int(all_labels[i].item())
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                lon, lat = rasterio.transform.xy(transform, cy, cx)
                features.append({
                    'type': 'Feature',
                    'geometry': {'type': 'Point',
                                 'coordinates': [lon, lat]},
                    'properties': {
                        'score':          round(score, 4),
                        'confidence_bin': score_to_bin(score),
                        'label':          label,
                        'pixel_x':        round(cx),
                        'pixel_y':        round(cy),
                        'box_width':      round(x2 - x1),
                        'box_height':     round(y2 - y1),
                    }
                })

        geojson = {
            'type': 'FeatureCollection',
            'crs':  {'type': 'name',
                     'properties': {'name': crs.to_string()}},
            'features': features
        }
        with open(job_dir / 'result.geojson', 'w') as f:
            json.dump(geojson, f, indent=2)

        # 7. Bake detections onto preview and save
        update_status(job_dir, 'running', 97, 'Rendering result preview...')
        result_img = bake_detections(preview_img, features, scale)
        result_img.save(job_dir / 'preview_result.png', format='PNG')

        # 8. Done
        del model
        gc.collect()
        update_status(job_dir, 'done', 100,
                      f'{n_det} ships detected')

    except Exception as e:
        import traceback
        update_status(job_dir, 'error', 0, str(e))