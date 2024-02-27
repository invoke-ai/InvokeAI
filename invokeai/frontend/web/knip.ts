import type { KnipConfig } from 'knip';

const config: KnipConfig = {
  ignore: [
    // This file is only used during debugging
    'src/app/store/middleware/debugLoggerMiddleware.ts',
  ],
  ignoreDependencies: ['@storybook/addon-docs', '@storybook/blocks', '@storybook/test', 'public/.*'],
  ignoreBinaries: ['only-allow'],
};

export default config;
