/**
 * api.js — all fetch calls to the backend
 */
const API = {

  async listJobs() {
    const r = await fetch('/api/jobs');
    return r.json();
  },

  async getStatus(id) {
    const r = await fetch('/api/jobs/' + id + '/status');
    return { ok: r.ok, data: r.ok ? await r.json() : null };
  },

  async deleteJob(id) {
    return fetch('/api/jobs/' + id, { method: 'DELETE' });
  },

  async setTTL(id, ttl) {
    return fetch('/api/jobs/' + id + '/ttl?ttl_minutes=' + ttl, { method: 'PATCH' });
  },

  previewUrl(id) {
    return '/api/jobs/' + id + '/preview';
  },

  previewResultUrl(id) {
    return '/api/jobs/' + id + '/preview_result';
  },

  resultUrl(id) {
    return '/api/jobs/' + id + '/result';
  },

  /**
   * Submit job with XHR for upload progress.
   * onProgress(loaded, total) called during upload.
   * Returns job_id string.
   */
  submitJob(formData, onProgress) {
    return new Promise(function(resolve, reject) {
      const xhr = new XMLHttpRequest();
      xhr.upload.onprogress = function(e) {
        if (e.lengthComputable) onProgress(e.loaded, e.total);
      };
      xhr.onload = function() {
        if (xhr.status === 200) {
          resolve(JSON.parse(xhr.responseText).job_id);
        } else {
          reject(new Error('HTTP ' + xhr.status));
        }
      };
      xhr.onerror = function() { reject(new Error('Network error')); };
      xhr.open('POST', '/api/jobs');
      xhr.send(formData);
    });
  },
};
