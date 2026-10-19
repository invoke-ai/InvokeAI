/** Verify Windows taskkill's missing-process result maps to ESRCH, matching POSIX cleanup behavior. */

import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import process from 'node:process';
import { test } from 'node:test';

import { killPreview, spawnPreview } from './preview-server.mjs';

/** A process that exits immediately, so its pid is real but reaped by the time we signal it. */
const spawnExited = async () => {
  const child = spawn(process.execPath, ['-e', ''], { stdio: 'ignore' });
  await new Promise((resolve) => {
    child.once('exit', resolve);
  });
  return child.pid;
};

test('reports an already-exited process as ESRCH', async () => {
  const pid = await spawnExited();

  assert.throws(() => killPreview(pid, 'SIGTERM'), { code: 'ESRCH' });
});

test('starts a preview server without a shell and stops it again', async (t) => {
  // Use a separate port so parallel journeys cannot conflict under --strictPort.
  const port = 4199;
  // URL pathnames contain an invalid leading slash for Windows drive paths; use import.meta.dirname.
  const preview = spawnPreview({
    cwd: resolve(import.meta.dirname, '..'),
    port,
    stdio: ['ignore', 'ignore', 'pipe'],
  });

  t.after(() => {
    try {
      killPreview(preview.pid, 'SIGKILL');
    } catch (error) {
      if (error?.code !== 'ESRCH') {
        throw error;
      }
    }
  });

  // Verify the child is Vite itself: Windows .cmd shims fail without a shell, and shell wrappers obscure the
  // server PID.
  const spawned = await new Promise((resolve) => {
    preview.once('spawn', () => resolve(true));
    preview.once('error', () => resolve(false));
  });

  assert.equal(spawned, true, 'the preview server could not be spawned');
  assert.ok(preview.pid, 'the preview server has no pid');
});
