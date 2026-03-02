/**
 * jobs.js — right-hand jobs list panel
 */
const Jobs = (() => {

  function formatAge(ts) {
    const s = Math.floor(Date.now() / 1000 - ts);
    if (s < 60)   return s + 's ago';
    if (s < 3600) return Math.floor(s / 60) + 'm ago';
    return Math.floor(s / 3600) + 'h ago';
  }

  async function refresh() {
    let jobs;
    try { jobs = await API.listJobs(); }
    catch(e) { return; }

    const list = document.getElementById('jobsList');
    list.innerHTML = '';

    if (!jobs.length) {
      list.innerHTML = '<div class="jobs-empty">No jobs yet</div>';
      return;
    }

    const viewId = State.get('viewJobId');
    for (const job of jobs) {
      list.appendChild(_buildCard(job, job.job_id === viewId));
    }
  }

  function _buildCard(job, isViewing) {
    const card     = document.createElement('div');
    card.className = 'job-card' + (isViewing ? ' viewing' : '');

    const ttlLabel   = job.ttl_minutes === -1 ? '∞' : job.ttl_minutes + 'm';
    const progress   = job.progress || 0;
    const isDone     = job.status === 'done';
    const isRun      = job.status === 'running' || job.status === 'uploading';
    const statusText = isDone
      ? job.message
      : job.status + (isRun ? ' (' + progress + '%)' : '');

    card.innerHTML =
      '<div class="job-name" title="' + (job.filename || job.job_id) + '">' + (job.filename || job.job_id) + '</div>' +
      '<div class="job-meta">' +
        '<span>' + formatAge(job.created_at) + '</span>' +
        '<span>TTL: ' + ttlLabel + '</span>' +
      '</div>' +
      (isRun ? '<div class="job-mini-bar"><div class="job-mini-fill" style="width:' + progress + '%"></div></div>' : '') +
      '<div class="job-status-line s-' + job.status + '">' + statusText + '</div>' +
      '<div class="job-actions">' +
        (isDone ? '<button class="job-btn job-btn-dl" data-action="dl">↓ GeoJSON</button>' : '') +
        '<button class="job-btn job-btn-pin" data-action="pin">' + (job.ttl_minutes === -1 ? 'Unpin' : 'Pin ∞') + '</button>' +
        '<button class="job-btn job-btn-del" data-action="del">✕ Del</button>' +
      '</div>';

    card.querySelector('[data-action=del]').onclick = function(e) {
      e.stopPropagation();
      Sidebar.deleteJob(job.job_id);
    };
    card.querySelector('[data-action=pin]').onclick = function(e) {
      e.stopPropagation();
      API.setTTL(job.job_id, job.ttl_minutes === -1 ? 60 : -1).then(refresh);
    };
    if (isDone) {
      card.querySelector('[data-action=dl]').onclick = function(e) {
        e.stopPropagation();
        _download(job.job_id);
      };
    }

    card.addEventListener('click', function() { _selectJob(job); });
    return card;
  }

  async function _selectJob(job) {
    if (job.job_id === State.get('viewJobId')) return;

    document.getElementById('previewFilename').textContent = job.filename || '';
    document.getElementById('detCount').textContent =
      job.status === 'done' ? (job.message || '') : '';

    const result = await API.getStatus(job.job_id);
    if (!result.ok) return;

    // Load all meta fields into the sidebar form
    _loadJobToForm(job);

    Preview.setForJob(job.job_id, result.data);

    if ((result.data.status === 'running' || result.data.status === 'queued') &&
        job.job_id !== State.get('activeJobId')) {
      Poller.start(job.job_id, { previewOnly: true });
    }

    refresh();
  }

  // ── Load job meta into left sidebar form ───────────────
  function _loadJobToForm(job) {
    // params is the nested dict stored in meta.json
    // the job object from /api/jobs merges meta + status so
    // we may find fields at job.params.X or directly at job.X
    const p = job.params || {};

    // ── Top-level ──────────────────────────────────────
    _setVal('modelType',    p.model_type   || '');
    _setVal('polarization', p.polarization || '');
    _setVal('inputFormat',  p.input_format || '');
    _setVal('windowSize',   p.window_size  || 800);
    _setVal('ttlMinutes',   job.ttl_minutes !== undefined ? job.ttl_minutes : 60);

    // ── Detection sliders ──────────────────────────────
    _setSlider('overlapPct',  p.overlap_pct  !== undefined ? p.overlap_pct  : 0.25, 'overlapVal', function(v){ return Math.round(v*100)+'%'; });
    _setSlider('scoreThresh', p.score_thresh !== undefined ? p.score_thresh : 0.3,  'scoreVal',   function(v){ return (+v).toFixed(2); });
    _setSlider('nmsIou',      p.nms_iou      !== undefined ? p.nms_iou      : 0.3,  'nmsVal',     function(v){ return (+v).toFixed(2); });

    // ── Contrast ───────────────────────────────────────
    const c = p.contrast || {};
    const contrastEnabled = c.enabled !== false;

    _setCheck('contrastEnabled', contrastEnabled);
    document.getElementById('contrastParams').style.display = contrastEnabled ? 'block' : 'none';

    _setVal('contrastMethod', c.method || 'percentile');

    _setSlider('percLow',  c.percentile_low  !== undefined ? c.percentile_low  : 5,   'percLowV',  function(v){ return v+'%'; });
    _setSlider('percHigh', c.percentile_high !== undefined ? c.percentile_high : 95,  'percHighV', function(v){ return v+'%'; });
    _setSlider('gamma',    c.gamma           !== undefined ? c.gamma           : 1.0, 'gammaV',    function(v){ return (+v).toFixed(1); });

    _setCheck('clahe', c.clahe !== false);
  }

  // ── DOM helpers ────────────────────────────────────────
  function _setVal(id, val) {
    if (val === undefined || val === null || val === '') return;
    var el = document.getElementById(id);
    if (el) el.value = val;
  }

  function _setCheck(id, val) {
    var el = document.getElementById(id);
    if (el) el.checked = !!val;
  }

  function _setSlider(inputId, val, spanId, fmt) {
    if (val === undefined || val === null) return;
    var el   = document.getElementById(inputId);
    var span = document.getElementById(spanId);
    if (el)   el.value = val;
    if (span) span.textContent = fmt(val);
  }

  function _download(jobId) {
    var a      = document.createElement('a');
    a.href     = API.resultUrl(jobId);
    a.download = 'detections.geojson';
    a.click();
  }

  return { refresh };
})();
