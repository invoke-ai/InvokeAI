import { defineConfig } from 'vite';
import { appConfig } from './config/vite.app.config';
import { packageConfig } from './config/vite.package.config';

export default defineConfig(({ mode }) => {
  if (mode === 'package') {
    return packageConfig;
  }

  return appConfig;
});
