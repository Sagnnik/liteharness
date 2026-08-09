import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'path';

export default defineConfig({
  base: '/',
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
