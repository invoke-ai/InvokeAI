/**
 * Serve the dev app against the in-repo mock backend. The mock takes a free port unless INVOKEAI_MOCK_BACKEND_PORT is
 * set, and INVOKEAI_MOCK_BACKEND_PROFILE picks its fixtures. Remaining arguments go to `vite dev`, for example
 * `pnpm dev:mock --port 5273 --strictPort`.
 */

import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { constants } from 'node:os';
import { dirname, join } from 'node:path';
import process from 'node:process';

import { startMockBackend } from './mock-backend.mjs';

// Resolve via vite/package.json because the exports map hides bin/vite.js.
const viteBin = join(dirname(createRequire(import.meta.url).resolve('vite/package.json')), 'bin', 'vite.js');

const profile = process.env.INVOKEAI_MOCK_BACKEND_PROFILE ?? 'representative';
const backend = await startMockBackend(Number(process.env.INVOKEAI_MOCK_BACKEND_PORT ?? 0), { profile });

process.stdout.write(`Mock backend (${profile}) at ${backend.origin}\n`);

const vite = spawn(process.execPath, [viteBin, 'dev', ...process.argv.slice(2)], {
  env: { ...process.env, INVOKEAI_DEV_BACKEND: backend.origin },
  stdio: 'inherit',
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => vite.kill(signal));
}

vite.on('exit', async (code, signal) => {
  await backend.close();
  // Report a signal death the way a shell would, so scripted callers see it.
  process.exitCode = code ?? 128 + (constants.signals[signal] ?? 0);
});
