import { defineConfig } from 'vite';

import { appConfig } from './config/vite.app.config.mjs';
import { packageConfig } from './config/vite.package.config.mjs';

export default defineConfig(({ mode }) => {
  if (mode === 'package') {
    return packageConfig;
  }

  return appConfig;
});
