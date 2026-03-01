/**
 * poller.js — polling logic for active and preview-only jobs
 */
const Poller = (() => {

  function start(jobId, opts = {}) {
    State.clearPoll();
    const interval = setInterval(() => _tick(jobId, opts), 1500);
    State.set('pollInterval', interval);
  }

  async function _tick(jobId, opts) {
    const { ok, data } = await API.getStatus(jobId).catch(() => ({ ok: false, data: null }));

    if (!ok) {
      // 404 — job expired or deleted
      State.clearPoll();
      if (State.get('activeJobId') === jobId) {
        Sidebar.setStatus('Job expired or deleted.');
        Sidebar.resetRunUI();
        State.set('activeJobId', null);
      }
      return;
    }

    const isActive  = State.get('activeJobId') === jobId;
    const isViewing = State.get('viewJobId')    === jobId;
    const progress  = data.progress || 0;

    // Update sidebar progress only for the job we submitted
    if (isActive && !opts.previewOnly) {
      Sidebar.setProgress(progress);
      Sidebar.setStatus(data.message || '');
    }

    // Update preview if user is watching this job
    if (isViewing) {
      if (data.status === 'running' && progress >= 15) {
        // Only reload if we're currently showing a spinner
        const panel = document.getElementById('panel');
        if (panel.querySelector('.spinner') || panel.querySelector('.preview-empty')) {
          Preview.loadImage(jobId, false);
        }
      }
    }

    if (data.status === 'done') {
      State.clearPoll();

      if (isActive && !opts.previewOnly) {
        document.getElementById('abortBtn').style.display = 'none';
        Sidebar.setProgress(100, 'done');
        Sidebar.setStatus(data.message || '');
        Sidebar.resetRunUI();
        State.set('activeJobId', null);
      }

      if (isViewing) {
        document.getElementById('detCount').textContent = data.message || '';
        Preview.loadImage(jobId, true);
      }

      Jobs.refresh();

    } else if (data.status === 'error') {
      State.clearPoll();

      if (isActive && !opts.previewOnly) {
        Sidebar.setProgress(100, 'error');
        Sidebar.setStatus('Error: ' + (data.message || 'unknown'));
        Sidebar.resetRunUI();
        State.set('activeJobId', null);
      }

      if (isViewing) {
        Preview.showEmpty('Job failed: ' + (data.message || ''));
      }

      Jobs.refresh();
    }
  }

  return { start };
})();
