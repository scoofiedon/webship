/**
 * jobs.js — right-hand jobs list panel
 */
const Jobs = (() => {

  function formatAge(ts) {
    const s = Math.floor(Date.now() / 1000 - ts);
    if (s < 60)    return `${s}s ago`;
    if (s < 3600)  return `${Math.floor(s / 60)}m ago`;
    return `${Math.floor(s / 3600)}h ago`;
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
      const card = _buildCard(job, job.job_id === viewId);
      list.appendChild(card);
    }
  }

  function _buildCard(job, isViewing) {
    const card    = document.createElement('div');
    card.className = `job-card${isViewing ? ' viewing' : ''}`;

    const ttlLabel = job.ttl_minutes === -1 ? '∞' : `${job.ttl_minutes}m`;
    const progress = job.progress || 0;
    const isDone   = job.status === 'done';
    const isRun    = job.status === 'running' || job.status === 'uploading';

    const statusText = isDone
      ? job.message
      : `${job.status}${isRun ? ` (${progress}%)` : ''}`;

    card.innerHTML = `
      <div class="job-name" title="${job.filename || job.job_id}">${job.filename || job.job_id}</div>
      <div class="job-meta">
        <span>${formatAge(job.created_at)}</span>
        <span>TTL: ${ttlLabel}</span>
      </div>
      ${isRun ? `<div class="job-mini-bar"><div class="job-mini-fill" style="width:${progress}%"></div></div>` : ''}
      <div class="job-status-line s-${job.status}">${statusText}</div>
      <div class="job-actions">
        ${isDone ? `<button class="job-btn job-btn-dl"  data-action="dl">↓ GeoJSON</button>` : ''}
        <button class="job-btn job-btn-pin" data-action="pin">
          ${job.ttl_minutes === -1 ? 'Unpin' : 'Pin ∞'}
        </button>
        <button class="job-btn job-btn-del" data-action="del">✕ Del</button>
      </div>`;

    // Button actions (stopPropagation so card click doesn't also fire)
    card.querySelector('[data-action=del]').onclick = e => {
      e.stopPropagation();
      Sidebar.deleteJob(job.job_id);
    };
    card.querySelector('[data-action=pin]').onclick = e => {
      e.stopPropagation();
      API.setTTL(job.job_id, job.ttl_minutes === -1 ? 60 : -1).then(refresh);
    };
    if (isDone) {
      card.querySelector('[data-action=dl]').onclick = e => {
        e.stopPropagation();
        _download(job.job_id);
      };
    }

    // Click card → switch preview
    card.addEventListener('click', () => _selectJob(job));

    return card;
  }

  async function _selectJob(job) {
    if (job.job_id === State.get('viewJobId')) return;

    // Update topbar
    document.getElementById('previewFilename').textContent = job.filename || '';
    document.getElementById('detCount').textContent =
      job.status === 'done' ? (job.message || '') : '';

    // Fetch fresh status
    const { ok, data } = await API.getStatus(job.job_id);
    if (!ok) return;

    Preview.setForJob(job.job_id, data);

    // If it's still running and not already being polled, start polling for preview updates
    if ((data.status === 'running' || data.status === 'queued') &&
        job.job_id !== State.get('activeJobId')) {
      Poller.start(job.job_id, { previewOnly: true });
    }

    refresh();
  }

  function _download(jobId) {
    const a    = document.createElement('a');
    a.href     = API.resultUrl(jobId);
    a.download = 'detections.geojson';
    a.click();
  }

  return { refresh };
})();
