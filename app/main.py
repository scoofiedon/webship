import os, uuid, json, asyncio, time, shutil, io
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import aiofiles
import redis
from rq import Queue

JOBS_DIR        = Path(os.environ.get('JOBS_DIR', '/jobs'))
JOB_TTL_MINUTES = int(os.environ.get('JOB_TTL_MINUTES', 60))
JOBS_DIR.mkdir(exist_ok=True)

r = redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379'))
q = Queue('inference', connection=r)


async def cleanup_daemon():
    while True:
        await asyncio.sleep(300)
        now = time.time()
        for job_dir in JOBS_DIR.iterdir():
            if not job_dir.is_dir():
                continue
            try:
                meta_path = job_dir / 'meta.json'
                if not meta_path.exists():
                    if now - job_dir.stat().st_mtime > 3600:
                        shutil.rmtree(job_dir, ignore_errors=True)
                    continue
                with open(meta_path) as f:
                    meta = json.load(f)
                ttl = meta.get('ttl_minutes', JOB_TTL_MINUTES)
                if ttl == -1:
                    continue  # infinite
                if now - meta['created_at'] > ttl * 60:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    print(f'Cleaned up job: {job_dir.name}')
            except Exception as e:
                print(f'Cleanup error: {e}')


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(cleanup_daemon())
    yield
    task.cancel()


app = FastAPI(title='SAR Ship Detector', lifespan=lifespan)
app.mount('/static', StaticFiles(directory='static'), name='static')


@app.get('/api/jobs')
async def list_jobs():
    """List all jobs with their status and metadata."""
    jobs = []
    for job_dir in sorted(JOBS_DIR.iterdir(),
                          key=lambda p: p.stat().st_mtime, reverse=True):
        if not job_dir.is_dir():
            continue
        try:
            meta_path   = job_dir / 'meta.json'
            status_path = job_dir / 'status.json'
            if not meta_path.exists():
                continue
            with open(meta_path)   as f: meta   = json.load(f)
            
            # Handle status.json with error tolerance
            status = {'status': 'unknown', 'progress': 0, 'message': ''}
            if status_path.exists():
                try:
                    with open(status_path) as f:
                        status_content = f.read().strip()
                        if status_content:
                            status = json.loads(status_content)
                except (json.JSONDecodeError, IOError):
                    # If status.json is corrupted, use default status
                    pass
            jobs.append({
                'job_id':     job_dir.name,
                'filename':   meta.get('filename', ''),
                'created_at': meta.get('created_at', 0),
                'ttl_minutes': meta.get('ttl_minutes', JOB_TTL_MINUTES),
                'params':     meta.get('params', {}),
                **status,
            })
        except Exception:
            continue
    return jobs


@app.post('/api/jobs')
async def submit_job(
    request:          Request,
    file:             UploadFile = File(...),
    model_type:       str   = Form('yolo'),
    polarization:     str   = Form('VV'),
    input_format:     str   = Form('byte'),
    window_size:      int   = Form(800),
    overlap_pct:      float = Form(0.25),
    score_thresh:     float = Form(0.3),
    nms_iou:          float = Form(0.3),
    ttl_minutes:      int   = Form(60),       # -1 = infinite
    contrast_enabled: bool  = Form(True),
    contrast_method:  str   = Form('percentile'),
    percentile_low:   float = Form(2.0),
    percentile_high:  float = Form(98.0),
    gamma:            float = Form(1.0),
    clahe:            bool  = Form(True),
    target_size:      int   = Form(800),
    # Traditional detection parameters
    pfa:              float = Form(1e-6),
    guard:            int   = Form(4),
    train:            int   = Form(16),
    buffer_deg:       float = Form(0.002),
    extra_dilation_px: int  = Form(3),
):
    job_id  = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir()

    # Stream upload
    image_path    = job_dir / file.filename
    total_size    = int(request.headers.get('content-length', 0))
    uploaded_size = 0

    async with aiofiles.open(image_path, 'wb') as out:
        while chunk := await file.read(1024 * 1024):
            await out.write(chunk)
            uploaded_size += len(chunk)
            progress = int(uploaded_size / total_size * 10) if total_size else 0
            with open(job_dir / 'status.json', 'w') as f:
                json.dump({'status': 'uploading', 'progress': progress,
                           'message': f'Uploading {uploaded_size//1024//1024}MB...'}, f)

    params = {
        'model_type': model_type,
        'window_size': window_size, 'overlap_pct': overlap_pct,
        'score_thresh': score_thresh, 'nms_iou': nms_iou,
        'polarization': polarization, 'input_format': input_format,
        'target_size': target_size,
        'device': 'cpu',
        'contrast': {
            'enabled': contrast_enabled, 'method': contrast_method,
            'percentile_low': percentile_low, 'percentile_high': percentile_high,
            'gamma': gamma, 'clahe': clahe,
        }
    }
    
    # Add traditional detection parameters if using traditional method
    if model_type == 'traditional':
        params.update({
            'pfa': pfa,
            'guard': guard,
            'train': train,
            'buffer_deg': buffer_deg,
            'extra_dilation_px': extra_dilation_px,
            'gshhg_path': './gshhg/GSHHS_f_L1.shp'  # Path to GSHHG data - REQUIRED for production
        })

    with open(job_dir / 'meta.json', 'w') as f:
        json.dump({
            'created_at':  time.time(),
            'filename':    file.filename,
            'ttl_minutes': ttl_minutes,
            'params':      params,
        }, f)

    with open(job_dir / 'status.json', 'w') as f:
        json.dump({'status': 'queued', 'progress': 10,
                   'message': 'Queued...'}, f)

    q.enqueue(
        'inference_jobs.run_inference_job',
        job_id, str(image_path), model_type, params,
        job_timeout=3600, job_id=job_id,
    )
    return {'job_id': job_id}


@app.get('/api/jobs/{job_id}/status')
async def job_status(job_id: str):
    p = JOBS_DIR / job_id / 'status.json'
    if not p.exists():
        return JSONResponse({'error': 'Not found'}, status_code=404)
    
    try:
        # Read file with retry logic to handle race conditions
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(p, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(0.1)
                            continue
                        else:
                            return JSONResponse({'error': 'Status file is empty'}, status_code=500)
                    return json.loads(content)
            except (json.JSONDecodeError, IOError) as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1)
                    continue
                else:
                    return JSONResponse({'error': f'Error reading status: {str(e)}'}, status_code=500)
    except Exception as e:
        return JSONResponse({'error': f'Unexpected error: {str(e)}'}, status_code=500)


@app.get('/api/jobs/{job_id}/preview')
async def get_preview(job_id: str):
    p = JOBS_DIR / job_id / 'preview.png'
    if not p.exists():
        return JSONResponse({'error': 'Not found'}, status_code=404)
    return FileResponse(p, media_type='image/png',
                        headers={'Cache-Control': 'no-cache'})


@app.get('/api/jobs/{job_id}/preview_result')
async def get_preview_result(job_id: str):
    p = JOBS_DIR / job_id / 'preview_result.png'
    if not p.exists():
        return JSONResponse({'error': 'Not found'}, status_code=404)
    return FileResponse(p, media_type='image/png',
                        headers={'Cache-Control': 'no-cache'})


@app.get('/api/jobs/{job_id}/result')
async def download_result(job_id: str):
    p = JOBS_DIR / job_id / 'result.geojson'
    if not p.exists():
        return JSONResponse({'error': 'Not found'}, status_code=404)
    return FileResponse(p, filename='detections.geojson',
                        media_type='application/geo+json')


@app.patch('/api/jobs/{job_id}/ttl')
async def update_ttl(job_id: str, ttl_minutes: int):
    meta_path = JOBS_DIR / job_id / 'meta.json'
    if not meta_path.exists():
        return JSONResponse({'error': 'Not found'}, status_code=404)
    with open(meta_path) as f:
        meta = json.load(f)
    meta['ttl_minutes'] = ttl_minutes
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    return {'ttl_minutes': ttl_minutes}


@app.delete('/api/jobs/{job_id}')
async def delete_job(job_id: str):
    from rq.job import Job
    try:
        Job.fetch(job_id, connection=r).cancel()
    except Exception:
        pass
    shutil.rmtree(JOBS_DIR / job_id, ignore_errors=True)
    return {'status': 'deleted'}


@app.get('/')
async def serve_ui():
    return FileResponse('static/index.html')