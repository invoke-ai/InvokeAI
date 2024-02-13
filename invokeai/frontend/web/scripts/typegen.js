import fs from 'node:fs';

import openapiTS from 'openapi-typescript';

const OPENAPI_URL = 'http://127.0.0.1:9090/openapi.json';
const OUTPUT_FILE = 'src/services/api/schema.ts';

async function main() {
  process.stdout.write(`Generating types "${OPENAPI_URL}" --> "${OUTPUT_FILE}"...`);
  const types = await openapiTS(OPENAPI_URL, {
    exportType: true,
    transform: (schemaObject) => {
      if ('format' in schemaObject && schemaObject.format === 'binary') {
        return schemaObject.nullable ? 'Blob | null' : 'Blob';
      }
    },
  });
  fs.writeFileSync(OUTPUT_FILE, types);
  process.stdout.write(`\nOK!\r\n`);
}

main();
