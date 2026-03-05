/**
 * preview.js — preview panel management
 */
const Preview = (() => {

  function _panel() { return document.getElementById('panel'); }

  function showEmpty(msg = 'Отправьте задачу для просмотра') {
    _panel().innerHTML = `
      <div class="preview-empty">
        <span class="preview-empty-icon">🛰</span>
        ${msg}
      </div>`;
  }

  function showSpinner(msg = 'Обработка...') {
    _panel().innerHTML = `
      <div class="preview-empty">
        <div class="spinner"></div>
        ${msg}
      </div>`;
  }

  /**
   * Load image for jobId (raw or with detections).
   * Guards against stale loads: each call stamps a loadToken;
   * if the viewJobId changes before the image loads, it's discarded.
   */
  function loadImage(jobId, withDetections, attempt = 0) {
    // Stamp this load attempt
    const token = `${jobId}-${withDetections}-${Date.now()}`;
    State.set('previewLoadId', token);

    const url = withDetections
      ? API.previewResultUrl(jobId)
      : API.previewUrl(jobId);

    const img = new Image();
    img.onload = () => {
      // Discard if job or mode changed while loading
      if (State.get('previewLoadId') !== token) return;
      if (State.get('viewJobId')     !== jobId)  return;

      const panel = _panel();
      panel.innerHTML = '';

      const el       = document.createElement('img');
      el.className   = 'preview-img';
      el.src         = img.src;
      panel.appendChild(el);

      _buildOverlay(panel, jobId, withDetections);
    };

    img.onerror = () => {
      if (State.get('viewJobId') !== jobId) return;
      if (attempt < 8) {
        setTimeout(() => loadImage(jobId, withDetections, attempt + 1), 2000);
      } else {
        showEmpty('Предпросмотр недоступен');
      }
    };

    img.src = `${url}?t=${Date.now()}`;
  }

  function _buildOverlay(panel, jobId, withDetections) {
    const ov = document.createElement('div');
    ov.className = 'preview-overlay';
    ov.innerHTML = `
      <button class="pill ${withDetections ? 'active' : ''}"  id="pillDet">Обнаружения</button>
      <button class="pill ${!withDetections ? 'active' : ''}" id="pillRaw">Исходное</button>`;
    panel.appendChild(ov);

    document.getElementById('pillDet').onclick = () => toggle(jobId, true);
    document.getElementById('pillRaw').onclick = () => toggle(jobId, false);
  }

  function toggle(jobId, det) {
    if (State.get('showDetections') === det) return;
    State.set('showDetections', det);
    loadImage(jobId, det);
  }

  function setForJob(jobId, statusData) {
    State.setViewJobId(jobId);
    const s = statusData.status;

    if (s === 'done') {
      State.set('showDetections', true);
      loadImage(jobId, true);
    } else if ((s === 'running' || s === 'uploading') &&
               (statusData.progress || 0) >= 15) {
      loadImage(jobId, false);
    } else if (s === 'queued' || s === 'uploading') {
      showSpinner('В очереди — предпросмотр скоро появится...');
    } else if (s === 'running') {
      showSpinner('Обработка...');
    } else if (s === 'error') {
      showEmpty('Задача не выполнена');
    } else {
      showEmpty();
    }
  }

  return { showEmpty, showSpinner, loadImage, setForJob, toggle };
})();
