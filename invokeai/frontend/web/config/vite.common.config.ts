import { UserConfig } from 'vite';

export const commonConfig: UserConfig = {
  base: './',
  server: {
    // Proxy HTTP requests to the flask server
    proxy: {
      '/outputs': {
        target: 'http://127.0.0.1:9090/outputs',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/outputs/, ''),
      },
      '/upload': {
        target: 'http://127.0.0.1:9090/upload',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/upload/, ''),
      },
      '/flaskwebgui-keep-server-alive': {
        target: 'http://127.0.0.1:9090/flaskwebgui-keep-server-alive',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/flaskwebgui-keep-server-alive/, ''),
      },
      // Proxy socket.io to the flask-socketio server
      '/socket.io': {
        target: 'ws://127.0.0.1:9090',
        ws: true,
      },
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
  build: {
    chunkSizeWarningLimit: 1500,
  },
};
