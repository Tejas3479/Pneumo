import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/static/',
  build: {
    outDir: '../app/static',
    emptyOutDir: false, // Keep ohif fallback folder intact
  },
  server: {
    proxy: {
      '/predict': 'http://localhost:8000',
      '/feedback': 'http://localhost:8000',
      '/fairness-audit': 'http://localhost:8000',
      '/run-federated-round': 'http://localhost:8000',
      '/studies': 'http://localhost:8000',
      '/metrics/drift': 'http://localhost:8000',
      '/audit-ledger/verify': 'http://localhost:8000',
      '/regulatory/model-card': 'http://localhost:8000',
      '/result': 'http://localhost:8000',
      '/dicomweb': 'http://localhost:8000',
    }
  }
})
