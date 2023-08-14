import fs from 'node:fs';
import openapiTS from 'openapi-typescript';
import { COLORS } from './colors.js';

const OPENAPI_URL = 'http://127.0.0.1:9090/openapi.json';
const OUTPUT_FILE = 'src/services/api/schema.d.ts';

async function main() {
  process.stdout.write(
    `Generating types "${OPENAPI_URL}" --> "${OUTPUT_FILE}"...\n\n`
  );
  const types = await openapiTS(OPENAPI_URL, {
    exportType: true,
    transform: (schemaObject, metadata) => {
      if ('format' in schemaObject && schemaObject.format === 'binary') {
        return schemaObject.nullable ? 'Blob | null' : 'Blob';
      }

      /**
       * Because invocations may have required fields that accept connection input, the generated
       * types may be incorrect.
       *
       * For example, the ImageResizeInvocation has a required `image` field, but because it accepts
       * connection input, it should be optional on instantiation of the field.
       *
       * To handle this, the schema exposes an `input` property that can be used to determine if the
       * field accepts connection input. If it does, we can make the field optional.
       */

      // Check if we are generating types for an invocation
      const isInvocationPath = metadata.path.match(
        /^#\/components\/schemas\/\w*Invocation$/
      );

      const hasInvocationProperties =
        schemaObject.properties &&
        ['id', 'is_intermediate', 'type'].every(
          (prop) => prop in schemaObject.properties
        );

      if (isInvocationPath && hasInvocationProperties) {
        // We only want to make fields optional if they are required
        if (!Array.isArray(schemaObject?.required)) {
          schemaObject.required = ['id', 'type'];
          return;
        }

        schemaObject.required.forEach((prop) => {
          const acceptsConnection = ['any', 'connection'].includes(
            schemaObject.properties?.[prop]?.['input']
          );

          if (acceptsConnection) {
            // remove this prop from the required array
            const invocationName = metadata.path.split('/').pop();
            console.log(
              `Making connectable field optional: ${COLORS.fg.green}${invocationName}.${COLORS.fg.cyan}${prop}${COLORS.reset}`
            );
            schemaObject.required = schemaObject.required.filter(
              (r) => r !== prop
            );
          }
        });

        schemaObject.required = [
          ...new Set(schemaObject.required.concat(['id', 'type'])),
        ];

        return;
      }
      // if (
      //   'input' in schemaObject &&
      //   (schemaObject.input === 'any' || schemaObject.input === 'connection')
      // ) {
      //   schemaObject.required = false;
      // }
    },
  });
  fs.writeFileSync(OUTPUT_FILE, types);
  process.stdout.write(`\nOK!\r\n`);
}

main();
