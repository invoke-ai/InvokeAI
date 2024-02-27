import type { KnipConfig } from 'knip';

const config: KnipConfig = {
  ignore: [
    // This file is only used during debugging
    'src/app/store/middleware/debugLoggerMiddleware.ts',
    // These are old schemas, used in migrations. Needs cleanup.
    'src/features/nodes/types/v2/**/*',
    'src/features/nodes/types/v1/**/*',
    // We don't want to check the public folder - contains images and translations
    'public/**/*',
  ],
  compilers: {
    //
    svg: () => '',
  },
};

export default config;
