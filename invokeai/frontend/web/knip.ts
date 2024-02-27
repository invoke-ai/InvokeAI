import type { KnipConfig } from 'knip';

const config: KnipConfig = {
  ignore: [
    // This file is only used during debugging
    'src/app/store/middleware/debugLoggerMiddleware.ts',
  ],
  ignoreBinaries: ['only-allow'],
};

export default config;
