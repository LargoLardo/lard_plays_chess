import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/engine": {
        target: "http://localhost:5000",
        changeOrigin: true,
        secure: false,
      },
      "/checkpoints": {
        target: "http://localhost:5000",
        changeOrigin: true,
        secure: false,
      },
      "/health": {
        target: "http://localhost:5000",
        changeOrigin: true,
        secure: false,
      },
    },
  }
})
