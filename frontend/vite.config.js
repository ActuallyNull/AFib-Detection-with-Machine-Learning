import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
const API_BASE_URL = import.meta.env.VITE_API_URL;

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: `http://${API_BASE_URL}`,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})

