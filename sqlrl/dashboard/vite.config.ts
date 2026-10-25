import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import { tanstackRouter } from '@tanstack/router-plugin/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiPath = env.VITE_API_URL?.startsWith('/') ? env.VITE_API_URL : '/api'

  return {
    plugins: [
      // Generates src/routeTree.gen.ts from src/routes and code-splits each route.
      tanstackRouter({ target: 'react', autoCodeSplitting: true }),
      react(),
      tailwindcss(),
    ],
    resolve: {
      alias: { '@': path.resolve(import.meta.dirname, './src') },
    },
    server: {
      // Same-origin API in development: cookies just work and no CORS is needed.
      proxy: { [apiPath]: { target: env.API_PROXY_TARGET ?? 'http://localhost:8000', changeOrigin: true } },
    },
    build: {
      target: 'es2022',
      sourcemap: true,
    },
  }
})
