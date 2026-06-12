import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/predict': 'http://localhost:8000',
      '/heatmap': 'http://localhost:8000',
      '/simulate': 'http://localhost:8000'
    }
  },
  build: {
    outDir: '../app/static',
    emptyOutDir: true
  },
  base: '/'
})
