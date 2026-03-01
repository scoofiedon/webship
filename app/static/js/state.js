/**
 * state.js — single source of truth for app state
 */
const State = (() => {
  const _s = {
    // Job being tracked for sidebar progress (just submitted)
    activeJobId:    null,
    // Job currently shown in preview panel
    viewJobId:      localStorage.getItem('viewJobId') || null,
    // Polling interval handle
    pollInterval:   null,
    // Whether preview shows detections or raw
    showDetections: true,
    // File selected in drop zone
    currentFile:    null,
    // Prevent concurrent preview loads for stale jobs
    previewLoadId:  null,
  };

  return {
    get: k      => _s[k],
    set: (k, v) => { _s[k] = v; },

    setViewJobId(id) {
      _s.viewJobId = id;
      if (id) localStorage.setItem('viewJobId', id);
      else    localStorage.removeItem('viewJobId');
    },

    clearPoll() {
      if (_s.pollInterval) {
        clearInterval(_s.pollInterval);
        _s.pollInterval = null;
      }
    },
  };
})();
