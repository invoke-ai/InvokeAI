import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const CLI = resolve(PACKAGE_ROOT, 'scripts/export-graphs.mjs');
const CAPABILITIES = resolve(PACKAGE_ROOT, 'src/features/generation/core/__fixtures__/architectureCapabilities.json');

const runCli = async (context, requestText) => {
  const directory = await mkdtemp(resolve(tmpdir(), 'invokeai-graphs-export-'));
  context.after(() => rm(directory, { force: true, recursive: true }));
  const requestPath = resolve(directory, 'request.json');
  const responsePath = resolve(directory, 'response.json');
  await writeFile(requestPath, requestText);

  const { code, stderr } = await promisify(execFile)(process.execPath, [CLI, requestPath, responsePath], {
    cwd: PACKAGE_ROOT,
  }).then(
    ({ stderr }) => ({ code: 0, stderr }),
    (error) => ({ code: error.code, stderr: error.stderr })
  );
  const response = await access(responsePath).then(
    async () => JSON.parse(await readFile(responsePath, 'utf8')),
    () => null
  );

  return { code, response, stderr };
};

test('writes per-case results through the real module graph and exits 0', async (context) => {
  const capabilities = JSON.parse(await readFile(CAPABILITIES, 'utf8'));
  const sdxl = { base: 'sdxl', format: 'diffusers', hash: 'blake3:sdxl', key: 'sdxl', name: 'SDXL', type: 'main' };
  const testCase = { height: 1024, mode: 'txt2img', positive_prompt: 'a fox', seed: 7, steps: 4, width: 1024 };
  const request = {
    capabilities,
    cases: [
      { ...testCase, id: 'ok', model_key: 'sdxl' },
      { ...testCase, id: 'missing', model_key: 'missing' },
    ],
    models: [sdxl],
    version: 1,
  };

  const { code, response } = await runCli(context, JSON.stringify(request));

  assert.equal(code, 0);
  assert.equal(response.version, 1);
  assert.deepEqual(
    response.cases.map((result) => [result.id, 'batch' in result]),
    [
      ['ok', true],
      ['missing', false],
    ]
  );
  assert.match(response.cases[1].error, /model_key "missing"/);
});

for (const [label, requestText] of [
  ['unparsable JSON', '{ "version": 1,'],
  ['a malformed request', JSON.stringify({ cases: [], models: [], version: 1 })],
]) {
  test(`rejects ${label} with exit 2 and writes no response`, async (context) => {
    const { code, response, stderr } = await runCli(context, requestText);

    assert.equal(code, 2);
    assert.equal(response, null);
    assert.match(stderr, /^graphs:export: /);
  });
}
