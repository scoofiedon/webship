/**
 * sidebar.js — form, submit, progress, new/abort
 */

// ── Private helpers (defined before Sidebar object) ────
function _val(id) { return document.getElementById(id).value; }
function _mb(b)   { return (b / 1024 / 1024).toFixed(1); }

function _handleFile(file) {
  if (!file) return;
  State.set('currentFile', file);
  const dz = document.getElementById('dropZone');
  dz.classList.add('has-file');
  dz.querySelector('.drop-icon').textContent = '🖼️';
  document.getElementById('dropLabel').textContent = file.name;
  document.getElementById('runBtn').disabled = false;
}

// ── Public API ─────────────────────────────────────────
const Sidebar = {

  initDropZone() {
    const input = document.getElementById('fi');
    const dz    = document.getElementById('dropZone');

    input.addEventListener('change', e => _handleFile(e.target.files[0]));
    dz.addEventListener('click',     () => input.click());
    dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('over'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('over'));
    dz.addEventListener('drop', e => {
      e.preventDefault();
      dz.classList.remove('over');
      _handleFile(e.dataTransfer.files[0]);
    });
  },

  initContrastToggle() {
    document.getElementById('contrastEnabled').addEventListener('change', function() {
      document.getElementById('contrastParams').style.display =
        this.checked ? 'block' : 'none';
    });
  },

  setProgress(pct, mode = '') {
    const fill = document.getElementById('pFill');
    fill.style.width = pct + '%';
    fill.className   = 'progress-fill' + (mode ? ' ' + mode : '');
  },

  setStatus(msg) {
    document.getElementById('statusMsg').textContent = msg || '';
  },

  resetRunUI() {
    document.getElementById('runBtn').disabled        = false;
    document.getElementById('newBtn').disabled        = false;
    document.getElementById('abortBtn').style.display = 'none';
  },

  newJob() {
    State.clearPoll();
    State.set('activeJobId', null);
    State.set('currentFile', null);

    const dz = document.getElementById('dropZone');
    dz.classList.remove('has-file');
    dz.querySelector('.drop-icon').textContent       = '📡';
    document.getElementById('dropLabel').textContent  = 'Drop GeoTIFF or click to browse';
    document.getElementById('fi').value               = '';
    document.getElementById('runBtn').disabled        = true;
    document.getElementById('abortBtn').style.display = 'none';
    Sidebar.setProgress(0);
    Sidebar.setStatus('');
  },

  async submitJob() {
    const file = State.get('currentFile');
    if (!file) return;

    const fd = new FormData();
    fd.append('file',             file);
    fd.append('model_type',       _val('modelType'));
    fd.append('polarization',     _val('polarization'));
    fd.append('input_format',     _val('inputFormat'));
    fd.append('window_size',      _val('windowSize'));
    fd.append('overlap_pct',      _val('overlapPct'));
    fd.append('score_thresh',     _val('scoreThresh'));
    fd.append('nms_iou',          _val('nmsIou'));
    fd.append('ttl_minutes',      _val('ttlMinutes'));
    fd.append('contrast_enabled', document.getElementById('contrastEnabled').checked);
    fd.append('contrast_method',  _val('contrastMethod'));
    fd.append('percentile_low',   _val('percLow'));
    fd.append('percentile_high',  _val('percHigh'));
    fd.append('gamma',            _val('gamma'));
    fd.append('clahe',            document.getElementById('clahe').checked);

    document.getElementById('runBtn').disabled        = true;
    document.getElementById('newBtn').disabled        = false;
    document.getElementById('abortBtn').style.display = 'block';
    Sidebar.setProgress(0);
    Sidebar.setStatus('Uploading...');

    let jobId;
    try {
      jobId = await API.submitJob(fd, (loaded, total) => {
        Sidebar.setProgress(Math.round(loaded / total * 10));
        Sidebar.setStatus('Uploading ' + _mb(loaded) + ' / ' + _mb(total) + ' MB');
      });
    } catch(e) {
      Sidebar.setStatus('Upload failed: ' + e.message);
      Sidebar.resetRunUI();
      return;
    }

    State.set('activeJobId', jobId);
    document.getElementById('previewFilename').textContent = file.name;
    document.getElementById('detCount').textContent        = '';
    Preview.setForJob(jobId, { status: 'queued', progress: 0 });

    Poller.start(jobId);
    Jobs.refresh();
  },

  async abortJob() {
    State.clearPoll();
    const jobId = State.get('activeJobId');
    if (jobId) {
      await API.deleteJob(jobId);
      if (State.get('viewJobId') === jobId) {
        Preview.showEmpty('Job aborted');
        State.setViewJobId(null);
        document.getElementById('previewFilename').textContent = '';
        document.getElementById('detCount').textContent        = '';
      }
      State.set('activeJobId', null);
    }
    Sidebar.resetRunUI();
    Sidebar.setStatus('Aborted.');
    Sidebar.setProgress(0);
    Jobs.refresh();
  },

  async deleteJob(jobId) {
    await API.deleteJob(jobId);
    if (jobId === State.get('activeJobId')) {
      State.clearPoll();
      State.set('activeJobId', null);
      Sidebar.resetRunUI();
      Sidebar.setStatus('');
      Sidebar.setProgress(0);
    }
    if (jobId === State.get('viewJobId')) {
      State.setViewJobId(null);
      Preview.showEmpty();
      document.getElementById('previewFilename').textContent = '';
      document.getElementById('detCount').textContent        = '';
    }
    Jobs.refresh();
  },
};
