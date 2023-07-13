import { UserConfig } from 'vite';
import { commonPlugins } from './common';

export const appConfig: UserConfig = {
  base: './',
  plugins: [...commonPlugins],
  build: {
    chunkSizeWarningLimit: 1500,
  },
  server: {
    // Proxy HTTP requests to the flask server
    proxy: {
      // Proxy socket.io to the nodes socketio server
      '/ws/socket.io': {
        target: 'ws://127.0.0.1:9090',
        ws: true,
      },
      // Proxy openapi schema definiton
      '/openapi.json': {
        target: 'http://127.0.0.1:9090/openapi.json',
        rewrite: (path) => path.replace(/^\/openapi.json/, ''),
        changeOrigin: true,
      },
      // proxy nodes api
      '/api/v1': {
        target: 'http://127.0.0.1:9090/api/v1',
        rewrite: (path) => path.replace(/^\/api\/v1/, ''),
        changeOrigin: true,
      },
    },
  },
};
