import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import test from 'node:test';

import { executeProjectFileJourney } from './run-project-file-journey.mjs';

const delay = (durationMs) =>
  new Promise((resolve) => {
    setTimeout(resolve, durationMs);
  });

const createPreview = () => {
  const preview = new EventEmitter();

  preview.pid = 123;
  preview.stderr = new EventEmitter();

  return preview;
};

const createSuccessfulDependencies = (events, overrides = {}) => {
  const preview = createPreview();
  const dependencies = {
    createTempDirectory: () => {
      events.push('temp:create');
      return '/tmp/project-file-journey-test';
    },
    killProcessGroup: (_pid, signal) => {
      events.push(`preview:${signal}`);
      queueMicrotask(() => preview.emit('exit', 0, signal));
    },
    launchBrowser: () => ({
      close: () => {
        events.push('browser:close');
      },
    }),
    now: () => performance.now(),
    removeTempDirectory: () => {
      events.push('temp:remove');
    },
    runRoundTrip: () => ({ projectId: 'imported-project' }),
    spawnPreview: () => preview,
    startBackend: () => ({
      close: () => {
        events.push('backend:close');
      },
    }),
    waitForPreview: () => undefined,
    ...overrides,
  };

  return { dependencies, preview };
};

test('the deadline starts before setup and disposes a backend that resolves after timeout', async () => {
  const events = [];
  const { dependencies } = createSuccessfulDependencies(events, {
    startBackend: async () => {
      events.push('backend:start');
      await delay(25);

      return {
        close: () => {
          events.push('backend:close');
        },
      };
    },
  });
  const startedAt = performance.now();

  await assert.rejects(
    executeProjectFileJourney({ cleanupTimeoutMs: 50, dependencies, timeoutMs: 5 }),
    /exceeded its 0\.005-second timeout/
  );

  assert.ok(performance.now() - startedAt < 100);
  assert.deepEqual(events, ['temp:create', 'backend:start', 'backend:close', 'temp:remove']);
});

test('teardown attempts every owned resource and preserves the primary failure', async () => {
  const events = [];
  const { dependencies, preview } = createSuccessfulDependencies(events, {
    killProcessGroup: (_pid, signal) => {
      events.push(`preview:${signal}`);

      if (signal === 'SIGKILL') {
        queueMicrotask(() => preview.emit('exit', 0, signal));
      }
    },
    launchBrowser: () => ({
      close: () => {
        events.push('browser:close');
        throw new Error('browser cleanup failed');
      },
    }),
    removeTempDirectory: () => {
      events.push('temp:remove');
    },
    runRoundTrip: ({ contexts }) => {
      contexts.add({
        close: () => {
          events.push('context:close');
          throw new Error('context cleanup failed');
        },
      });
      throw new Error('primary journey failure');
    },
    startBackend: () => ({
      close: () => {
        events.push('backend:close');
        throw new Error('backend cleanup failed');
      },
    }),
  });

  await assert.rejects(executeProjectFileJourney({ cleanupTimeoutMs: 50, dependencies, timeoutMs: 500 }), (error) => {
    assert.match(error.message, /primary journey failure/);
    assert.match(error.message, /context cleanup failed/);
    assert.match(error.message, /browser cleanup failed/);
    assert.match(error.message, /backend cleanup failed/);
    return true;
  });

  assert.equal(events.includes('context:close'), true);
  assert.equal(events.includes('browser:close'), true);
  assert.equal(events.includes('preview:SIGTERM'), true);
  assert.equal(events.includes('preview:SIGKILL'), true);
  assert.equal(events.includes('backend:close'), true);
  assert.equal(events.at(-1), 'temp:remove');
});

test('a preview process error fails the journey and still tears down owned resources', async () => {
  const events = [];
  const { dependencies, preview } = createSuccessfulDependencies(events, {
    spawnPreview: () => {
      queueMicrotask(() => preview.emit('error', new Error('preview spawn failed')));
      return preview;
    },
  });

  await assert.rejects(
    executeProjectFileJourney({ cleanupTimeoutMs: 50, dependencies, timeoutMs: 500 }),
    /preview spawn failed/
  );

  assert.equal(events.includes('preview:SIGTERM'), true);
  assert.equal(events.includes('backend:close'), true);
  assert.equal(events.at(-1), 'temp:remove');
});

test('a browser error raised during teardown prevents a success result', async () => {
  const events = [];
  const { dependencies } = createSuccessfulDependencies(events, {
    launchBrowser: ({ errors }) => ({
      close: () => {
        events.push('browser:close');
        errors.push(new Error('late browser error'));
      },
    }),
  });

  await assert.rejects(
    executeProjectFileJourney({ cleanupTimeoutMs: 50, dependencies, timeoutMs: 500 }),
    /late browser error/
  );

  assert.equal(events.at(-1), 'temp:remove');
});
