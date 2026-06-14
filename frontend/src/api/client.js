import axios from 'axios';

export const getBaseURL = () => {
  try {
    const saved = localStorage.getItem('pneumodex_settings');
    if (saved) {
      const settings = JSON.parse(saved);
      return settings.serverUrl || '/';
    }
  } catch (e) {}
  return '/';
};

const client = axios.create({
  baseURL: getBaseURL(),
  timeout: 60000, // 60s request timeout (fairness audit and drift can be slow)
});

// Request interceptor: attach API key header only once
client.interceptors.request.use((config) => {
  try {
    const saved = localStorage.getItem('pneumodex_settings');
    if (saved) {
      const settings = JSON.parse(saved);
      if (settings.apiKey) {
        // Send only X-API-Key (not both — avoid leaking key in extra header)
        config.headers['X-API-Key'] = settings.apiKey;
      }
    }
  } catch (e) {}
  return config;
}, (error) => {
  return Promise.reject(error);
});

// Response interceptor: global error handling
client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      const status = error.response.status;
      if (status === 403) {
        console.error('[API] 403 Forbidden — check PNEUMO_API_KEY setting.');
      } else if (status === 503) {
        console.error('[API] 503 Service Unavailable — Redis/Celery may be offline.');
      } else if (status >= 500) {
        console.error(`[API] Server error ${status}:`, error.response.data?.detail || error.message);
      }
    } else if (error.code === 'ECONNABORTED') {
      console.error('[API] Request timed out after 60 seconds.');
    }
    return Promise.reject(error);
  }
);

export const updateBaseURL = (url) => {
  client.defaults.baseURL = url || '/';
};

export const api = {
  // Predict: upload chest X-ray for inference
  predict: async (file, modelType, saveImage = true) => {
    const formData = new FormData();
    formData.append('file', file);
    const mType = modelType || 'vit';
    const response = await client.post(
      `/predict?uncertainty=true&model_type=${mType}&save_image=${saveImage}`,
      formData
    );
    return response.data; // Returns { task_id, status }
  },

  // Feedback: submit clinician correction for active learning
  submitFeedback: async (imagePath, clinicianLabel) => {
    const response = await client.post('/feedback', {
      image_path: imagePath,
      clinician_label: clinicianLabel,
    });
    return response.data;
  },

  // DICOMweb QIDO-RS: list all studies (returns DICOM JSON format)
  getStudies: async () => {
    const response = await client.get('/dicomweb/studies');
    return response.data;
  },

  // Flat studies endpoint: simple list for the studies page table
  getStudiesList: async () => {
    const response = await client.get('/studies');
    return response.data;
  },

  // QIDO-RS series for a study
  getStudySeries: async (studyUid) => {
    const response = await client.get(`/dicomweb/studies/${studyUid}/series`);
    return response.data;
  },

  // QIDO-RS instances for a series
  getSeriesInstances: async (studyUid, seriesUid) => {
    const response = await client.get(`/dicomweb/studies/${studyUid}/series/${seriesUid}/instances`);
    return response.data;
  },

  // Study details (flat, with patient metadata)
  getStudyDetails: async (studyUid) => {
    const response = await client.get(`/studies/${studyUid}`);
    return response.data;
  },

  // Cached prediction for a study
  getStudyPrediction: async (studyUid) => {
    const response = await client.get(`/studies/${studyUid}/prediction`);
    return response.data;
  },

  // Enqueue prediction on a stored DICOM study
  predictStudy: async (studyUid, modelType) => {
    const mType = modelType || 'vit';
    const response = await client.post(`/studies/${studyUid}/predict?model_type=${mType}`);
    return response.data;
  },

  // Federated learning round
  runFederated: async () => {
    const response = await client.post('/run-federated-round');
    return response.data;
  },

  // Fairness audit (synchronous on backend — blocks up to 30s)
  getFairness: async () => {
    const response = await client.get('/fairness-audit', { timeout: 45000 });
    return response.data;
  },

  // Drift check: enqueues task, returns { task_id, status }
  getDrift: async () => {
    const response = await client.get('/metrics/drift');
    return response.data;
  },

  // Historical drift records from audit ledger
  getDriftHistory: async () => {
    const response = await client.get('/metrics/drift/history');
    return response.data;
  },

  // STOW-RS: upload DICOM file to the DICOMweb store
  // Sends as proper multipart/related; type="application/dicom"
  stowDicom: async (file) => {
    const arrayBuffer = await file.arrayBuffer();
    const boundary = `DICOMwebBoundary${Date.now()}`;
    // Build multipart/related body manually
    const header = `--${boundary}\r\nContent-Type: application/dicom\r\n\r\n`;
    const footer = `\r\n--${boundary}--`;
    const headerBytes = new TextEncoder().encode(header);
    const footerBytes = new TextEncoder().encode(footer);
    const fileBytes = new Uint8Array(arrayBuffer);
    const bodyBytes = new Uint8Array(headerBytes.length + fileBytes.length + footerBytes.length);
    bodyBytes.set(headerBytes, 0);
    bodyBytes.set(fileBytes, headerBytes.length);
    bodyBytes.set(footerBytes, headerBytes.length + fileBytes.length);

    const response = await client.post('/dicomweb/studies', bodyBytes, {
      headers: {
        'Content-Type': `multipart/related; type="application/dicom"; boundary=${boundary}`,
      },
    });
    return response.data;
  },

  // Active Learning: get flagged (uncertain) predictions for review
  getFlaggedSamples: async () => {
    const response = await client.get('/active-learning/flagged');
    return response.data;
  },

  // Active Learning: get status (counts + threshold)
  getActiveLearningStatus: async () => {
    const response = await client.get('/active-learning/status');
    return response.data;
  },

  // Active Learning: manually trigger fine-tuning
  triggerRetrain: async () => {
    const response = await client.post('/active-learning/trigger-retrain');
    return response.data;
  },

  // Regulatory: generate model_card.md — returns { task_id, status }
  createModelCard: async () => {
    const response = await client.post('/regulatory/model-card');
    return response.data;
  },

  // Audit ledger: verify hash chain integrity
  verifyLedger: async () => {
    const response = await client.get('/audit-ledger/verify');
    return response.data;
  },

  // Poll a Celery task result
  pollTask: async (taskId) => {
    const response = await client.get(`/result/${taskId}?format=json`);
    return response.data;
  },
};
