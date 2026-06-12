import axios from 'axios';

const api = axios.create({
  baseURL: '', // Uses relative path, leveraging Vite proxy in dev & same origin in prod
});

export const uploadCSV = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const getHeatmap = async () => {
  const response = await api.get('/heatmap');
  return response.data;
};

export const simulateDisruption = async (port, daysClosed) => {
  const response = await api.post('/simulate', {
    port,
    days_closed: daysClosed,
  });
  return response.data;
};
