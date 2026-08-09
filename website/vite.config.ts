import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'path';

export default defineConfig({
  // GitHub Pages serves this repository beneath /ness-agent/. Vite rewrites
  // imported content assets against this base instead of emitting root URLs.
  base: process.env.GITHUB_ACTIONS ? '/ness-agent/' : '/',
  plugins: [
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
      '@root': path.resolve(import.meta.dirname, '..'), // Access top-level repo files (README, CHANGELOG)
    },
  },
  server: {
    fs: {
      allow: ['..'],
    },
  },
});
