import type { ConfigFile } from '@rtk-query/codegen-openapi';

const config: ConfigFile = {
  schemaFile: 'https://petstore3.swagger.io/api/v3/openapi.json',
  apiFile: './src/app/emptyApi.ts',
  apiImport: 'emptySplitApi',
  outputFile: './src/app/invokeApi.ts',
  exportName: 'invokeApi',
  hooks: true,
};

export default config;
