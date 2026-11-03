/**
 * Run Vite's entry directly under Node: Windows cannot exec pnpm's .cmd shim without a shell, whose PID would
 * prevent reliable server-tree teardown.
 */

import { spawn, spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import process from 'node:process';

const isWindows = process.platform === 'win32';

// Resolve via vite/package.json because the exports map hides bin/vite.js.
const viteBin = join(dirname(createRequire(import.meta.url).resolve('vite/package.json')), 'bin', 'vite.js');

/** Bind preview to a fixed loopback port; forward other options to spawn. */
export const spawnPreview = ({ port, ...options }) =>
  spawn(process.execPath, [viteBin, 'preview', '--host', '127.0.0.1', '--port', String(port), '--strictPort'], {
    // Use a separate POSIX process group for tree teardown; on Windows detached would open a console instead.
    detached: !isWindows,
    ...options,
  });

/** Stop the preview process tree; preserve ESRCH for callers handling an already-exited server. */
export const killPreview = (pid, signal) => {
  if (!isWindows) {
    process.kill(-pid, signal);
    return;
  }

  // Windows needs taskkill /T /F; a graceful console event is unavailable without a shared console.
  const result = spawnSync('taskkill', ['/pid', String(pid), '/T', '/F'], { stdio: 'ignore' });

  // taskkill reports 128 for "no such process", which is what ESRCH means to every caller here.
  if (result.status === 128) {
    throw Object.assign(new Error(`No such process: ${pid}`), { code: 'ESRCH' });
  }

  if (result.error) {
    throw result.error;
  }
};
