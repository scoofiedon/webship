import os, uuid, json, asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import aiofiles
import redis
from rq import Queue

DATA_DIR  = Path(os.environ.get('DATA_DIR', '/data'))
DATA_DIR.mkdir(exist_ok=True)

r         = redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379'))
q         = Queue('inference', connection=r)

app       = FastAPI(title='SAR Ship Detector')
app.mount('/static', StaticFiles(directory='static'), name='static')


@app.post('/api/jobs')
async def submit_job(
    file:            UploadFile = File(...),
    model_type:      str = Form('yolo'),        # 'yolo' or 'fasterrcnn'
    polarization:    str = Form('VV'),
    input_format:    str = Form('byte'),
    window_size:     int = Form(800),
    overlap_pct:     float = Form(0.25),
    score_thresh:    float = Form(0.3),
    nms_iou:         float = Form(0.3),
    contrast_enabled: bool = Form(False),
    contrast_method:  str = Form('percentile'),
    percentile_low:   float = Form(2.0),
    percentile_high:  float = Form(98.0),
    gamma:            float = Form(1.0),
    clahe:            bool = Form(False),
):
    job_id  = str(uuid.uuid4())
    job_dir = DATA_DIR / job_id
    job_dir.mkdir()

    # Save uploaded file
    image_path = job_dir / file.filename
    async with aiofiles.open(image_path, 'wb') as f:
        await f.write(await file.read())

    params = {
        'window_size':  window_size,
        'overlap_pct':  overlap_pct,
        'score_thresh': score_thresh,
        'nms_iou':      nms_iou,
        'polarization': polarization,
        'input_format': input_format,
        'device':       'cpu',
        'contrast': {
            'enabled':         contrast_enabled,
            'method':          contrast_method,
            'percentile_low':  percentile_low,
            'percentile_high': percentile_high,
            'gamma':           gamma,
            'clahe':           clahe,
        }
    }

    # Write initial status
    with open(job_dir / 'status.json', 'w') as f:
        json.dump({'status': 'queued', 'progress': 0, 'message': ''}, f)

    # Enqueue job in worker
    q.enqueue(
        'inference_jobs.run_inference_job',
        job_id, str(image_path), model_type, params,
        job_timeout=3600,
        job_id=job_id,
    )

    return {'job_id': job_id}


@app.get('/api/jobs/{job_id}/status')
async def job_status(job_id: str):
    status_path = DATA_DIR / job_id / 'status.json'
    if not status_path.exists():
        return JSONResponse({'error': 'Job not found'}, status_code=404)
    with open(status_path) as f:
        return json.load(f)


@app.get('/api/jobs/{job_id}/result')
async def download_result(job_id: str, background_tasks: BackgroundTasks):
    result_path = DATA_DIR / job_id / 'result.geojson'
    if not result_path.exists():
        return JSONResponse({'error': 'Result not ready'}, status_code=404)

    # Schedule cleanup after download
    background_tasks.add_task(cleanup_job, job_id)
    return FileResponse(result_path, filename='detections.geojson',
                        media_type='application/geo+json')


@app.delete('/api/jobs/{job_id}')
async def abort_job(job_id: str):
    """Called when user closes browser or aborts."""
    cleanup_job(job_id)
    return {'status': 'deleted'}


def cleanup_job(job_id: str):
    import shutil
    job_dir = DATA_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)


@app.get('/')
async def serve_ui():
    return FileResponse('static/index.html')