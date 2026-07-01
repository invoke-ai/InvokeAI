/* eslint-disable no-console */
import fs from 'node:fs';

import openapiTS, { astToString } from 'openapi-typescript';
import ts from 'typescript';

const OPENAPI_URL = 'http://127.0.0.1:9090/openapi.json';
const OUTPUT_FILE = 'src/services/api/schema.ts';

async function generateTypes(schema) {
  process.stdout.write(`Generating types ${OUTPUT_FILE}...`);

  // Use https://ts-ast-viewer.com to figure out how to create these AST nodes - define a type and use the bottom-left pane's output
  // `Blob` type
  const BLOB = ts.factory.createTypeReferenceNode(ts.factory.createIdentifier('Blob'));
  // `null` type
  const NULL = ts.factory.createLiteralTypeNode(ts.factory.createNull());
  // `Record<string, unknown>` type
  const RECORD_STRING_UNKNOWN = ts.factory.createTypeReferenceNode(ts.factory.createIdentifier('Record'), [
    ts.factory.createKeywordTypeNode(ts.SyntaxKind.StringKeyword),
    ts.factory.createKeywordTypeNode(ts.SyntaxKind.UnknownKeyword),
  ]);

  const types = await openapiTS(schema, {
    exportType: true,
    transform: (schemaObject) => {
      if ('format' in schemaObject && schemaObject.format === 'binary') {
        return schemaObject.nullable ? ts.factory.createUnionTypeNode([BLOB, NULL]) : BLOB;
      }
      if (schemaObject.title === 'MetadataField') {
        // This is `Record<string, never>` by default, but it actually accepts any a dict of any valid JSON value.
        return RECORD_STRING_UNKNOWN;
      }
    },
    defaultNonNullable: false,
  });
  let output = astToString(types);

  // Post-process: openapi-typescript sometimes computes enum types from `const`
  // usage in discriminated unions rather than from the enum definition itself,
  // dropping values that only appear in some union members. Patch the generated
  // output to match the OpenAPI schema's actual enum definitions.
  //
  // The `schema` parameter is a parsed JSON object when piped from stdin, or
  // a URL/Buffer when passed as an argument. We only patch in the JSON case.
  if (schema && typeof schema === 'object' && !Buffer.isBuffer(schema)) {
    const schemas = schema.components?.schemas;
    if (schemas) {
      // Collect all string enum types and their expected values from the OpenAPI schema
      for (const [typeName, typeDef] of Object.entries(schemas)) {
        if (typeDef && typeDef.type === 'string' && Array.isArray(typeDef.enum)) {
          const expectedUnion = typeDef.enum.map((v) => `"${v}"`).join(' | ');
          // Match the type definition line. These appear as:
          //   `TypeName: "val1" | "val2" | ...;`
          // Use word boundary to avoid matching types that contain this
          // type name as a substring (e.g. ModelType vs BaseModelType).
          const regex = new RegExp(`(\\b${typeName}: )"[^;]+(;)`);
          const match = output.match(regex);
          if (match) {
            output = output.replace(regex, `$1${expectedUnion}$2`);
          }
        }
      }
    }
  }

  fs.writeFileSync(OUTPUT_FILE, output);
  process.stdout.write(`\nOK!\r\n`);
}

function main() {
  const encoding = 'utf-8';

  if (process.stdin.isTTY) {
    // Handle generating types with an arg (e.g. URL or path to file)
    if (process.argv.length > 3) {
      console.error('Usage: typegen.js <openapi.json>');
      process.exit(1);
    }
    if (process.argv[2]) {
      const schema = new Buffer.from(process.argv[2], encoding);
      generateTypes(schema);
    } else {
      generateTypes(OPENAPI_URL);
    }
  } else {
    // Handle generating types from stdin
    let schema = '';
    process.stdin.setEncoding(encoding);

    process.stdin.on('readable', function () {
      const chunk = process.stdin.read();
      if (chunk !== null) {
        schema += chunk;
      }
    });

    process.stdin.on('end', function () {
      generateTypes(JSON.parse(schema));
    });
  }
}

main();
