import axios from 'axios';

const client = axios.create({
  baseURL: '/', // Maps to proxy in dev, serves relative in production
});

export const api = {
  predict: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await client.post('/predict?uncertainty=true', formData);
    return response.data; // Returns { task_id, status }
  },

  submitFeedback: async (imagePath, clinicianLabel) => {
    const response = await client.post('/feedback', {
      image_path: imagePath,
      clinician_label: clinicianLabel,
    });
    return response.data;
  },

  getStudies: async () => {
    const response = await client.get('/dicomweb/studies');
    return response.data;
  },

  getStudySeries: async (studyUid) => {
    const response = await client.get(`/dicomweb/studies/${studyUid}/series`);
    return response.data;
  },

  getSeriesInstances: async (studyUid, seriesUid) => {
    const response = await client.get(`/dicomweb/studies/${studyUid}/series/${seriesUid}/instances`);
    return response.data;
  },

  getStudyPrediction: async (studyUid) => {
    const response = await client.get(`/studies/${studyUid}/prediction`);
    return response.data; // Returns cached prediction details
  },

  predictStudy: async (studyUid) => {
    const response = await client.post(`/studies/${studyUid}/predict`);
    return response.data; // Enqueues prediction, returns { task_id, status }
  },

  runFederated: async () => {
    const response = await client.post('/run-federated-round');
    return response.data;
  },

  getFairness: async () => {
    const response = await client.get('/fairness-audit');
    return response.data; // Synchronously blocks on Celery validation audit
  },

  getDrift: async () => {
    const response = await client.get('/metrics/drift');
    return response.data; // Enqueues drift task, returns { task_id, status }
  },

  createModelCard: async () => {
    const response = await client.post('/regulatory/model-card');
    return response.data; // Enqueues model card task, returns { task_id, status }
  },

  verifyLedger: async () => {
    const response = await client.get('/audit-ledger/verify');
    return response.data;
  },

  pollTask: async (taskId) => {
    const response = await client.get(`/result/${taskId}`);
    return response.data; // { status: 'PENDING' | 'SUCCESS' | 'FAILED', result?: any, error?: string }
  },
};
