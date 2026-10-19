import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import ts from 'typescript';

const generator = fileURLToPath(new URL('./typegen.js', import.meta.url));

test('generated contracts preserve upload, metadata, enum, and optional-default types', () => {
  const cwd = mkdtempSync(join(tmpdir(), 'invokeai-contracts-'));
  try {
    const schema = {
      openapi: '3.1.0',
      info: { title: 'Contract fixture', version: '1' },
      paths: {},
      components: {
        schemas: {
          Upload: { type: 'string', contentMediaType: 'application/octet-stream' },
          LegacyUpload: { type: 'string', format: 'binary', nullable: true },
          MetadataField: { type: 'object', title: 'MetadataField' },
          Mode: { type: 'string', enum: ['used', 'unused'] },
          Variant: { type: 'object', properties: { mode: { $ref: '#/components/schemas/Mode' } } },
          VariantUnion: {
            oneOf: [{ $ref: '#/components/schemas/Variant' }],
            discriminator: { propertyName: 'mode', mapping: { used: '#/components/schemas/Variant' } },
          },
          Defaulted: { type: 'object', properties: { count: { type: 'integer', default: 1 } } },
        },
      },
    };
    const result = spawnSync(process.execPath, [generator], { cwd, input: JSON.stringify(schema), encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
    const consumer = join(cwd, 'consumer.ts');
    writeFileSync(
      consumer,
      `import type { components } from './schema';
const upload: components['schemas']['Upload'] = new Blob();
const nullableUpload: components['schemas']['LegacyUpload'] = null;
const legacyUpload: components['schemas']['LegacyUpload'] = new Blob();
const metadata: components['schemas']['MetadataField'] = { arbitrary: { nested: true } };
const usedMode: components['schemas']['Mode'] = 'used';
const unusedMode: components['schemas']['Mode'] = 'unused';
const defaulted: components['schemas']['Defaulted'] = {};
// @ts-expect-error Uploads are binary, not strings.
const invalidUpload: components['schemas']['Upload'] = 'file contents';
// @ts-expect-error Modern uploads are not nullable without a nullable declaration.
const invalidNull: components['schemas']['Upload'] = null;
// @ts-expect-error Metadata must remain an object.
const invalidMetadata: components['schemas']['MetadataField'] = 'metadata';
// @ts-expect-error The enum must remain bounded.
const invalidMode: components['schemas']['Mode'] = 'other';
`
    );
    const program = ts.createProgram([consumer], { noEmit: true, strict: true, skipLibCheck: true });
    const diagnostics = ts.getPreEmitDiagnostics(program);
    assert.deepEqual(
      diagnostics.map((d) => ts.flattenDiagnosticMessageText(d.messageText, '\n')),
      []
    );
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
});

test('invalid input fails without replacing an existing contract', () => {
  const cwd = mkdtempSync(join(tmpdir(), 'invokeai-contracts-invalid-'));
  try {
    writeFileSync(join(cwd, 'schema.ts'), 'previous contract\n');
    const result = spawnSync(process.execPath, [generator], { cwd, input: '{invalid', encoding: 'utf8' });
    assert.notEqual(result.status, 0);
    assert.equal(readFileSync(join(cwd, 'schema.ts'), 'utf8'), 'previous contract\n');
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
});
