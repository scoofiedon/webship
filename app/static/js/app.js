/**
 * app.js — init and boot
 */
window.addEventListener('load', async () => {
  Sidebar.initDropZone();
  Sidebar.initContrastToggle();
  Sidebar.initModelTypeToggle();

  // Wire range inputs to their display spans
  _wireRange('overlapPct',  'overlapVal', v => (v * 100).toFixed(0) + '%');
  _wireRange('scoreThresh', 'scoreVal',   v => (+v).toFixed(2));
  _wireRange('nmsIou',      'nmsVal',     v => (+v).toFixed(2));
  _wireRange('percLow',     'percLowV',   v => v + '%');
  _wireRange('percHigh',    'percHighV',  v => v + '%');
  _wireRange('gamma',       'gammaV',     v => (+v).toFixed(1));
  
  // Traditional detection parameters
  _wireRange('pfa',              'pfaVal',     v => (+v).toExponential(1));
  _wireRange('guard',            'guardVal',   v => v);
  _wireRange('train',            'trainVal',   v => v);
  _wireRange('bufferMeters',       'bufferMetersVal', v => v);
  _wireRange('extraDilationPx', 'extraDilationPxVal', v => v);

  // Wire buttons
  document.getElementById('runBtn').onclick   = () => Sidebar.submitJob();
  document.getElementById('abortBtn').onclick = () => Sidebar.abortJob();
  document.getElementById('newBtn').onclick   = () => Sidebar.newJob();

  // Load jobs list
  await Jobs.refresh();

  // Restore previously viewed job
  const viewId = State.get('viewJobId');
  if (viewId) {
    const { ok, data } = await API.getStatus(viewId);
    if (!ok) {
      State.setViewJobId(null);
    } else {
      const jobs    = await API.listJobs().catch(() => []);
      const jobMeta = jobs.find(j => j.job_id === viewId);
      if (jobMeta) {
        document.getElementById('previewFilename').textContent = jobMeta.filename || '';
        document.getElementById('detCount').textContent =
          data.status === 'done' ? (data.message || '') : '';
      }
      Preview.setForJob(viewId, data);
      if (data.status === 'running' || data.status === 'queued') {
        Poller.start(viewId, { previewOnly: true });
      }
    }
  }

  // Refresh jobs list periodically
  setInterval(() => Jobs.refresh(), 8000);
});

function _wireRange(inputId, spanId, fmt) {
  const el   = document.getElementById(inputId);
  const span = document.getElementById(spanId);
  if (!el || !span) return;
  const update = () => { span.textContent = fmt(el.value); };
  el.addEventListener('input', update);
  update(); // init
}
