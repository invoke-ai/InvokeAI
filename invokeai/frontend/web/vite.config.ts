import { defineConfig } from 'vite';
import { commonConfig } from './config/vite.common.config';
import { appConfig } from './config/vite.app.config';
import { packageConfig } from './config/vite.package.config';
import { defaultsDeep } from 'lodash';

export default defineConfig(({ mode }) => {
  if (mode === 'package') {
    return defaultsDeep(packageConfig, commonConfig);
  }

  return defaultsDeep(appConfig, commonConfig);
});
