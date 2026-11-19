import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { DEFAULT_LOGGING_CONFIG, LOG_NAMESPACES, type LoggingConfig } from './contracts';
import {
  clearLogs,
  configureLogging,
  createLogger,
  getLoggingConfig,
  getLogSnapshot,
  LOG_RETENTION_LIMITS,
  recordTiming,
  resetLogging,
  subscribeLogs,
} from './logger';

const flushMicrotasks = () =>
  new Promise<void>((resolve) => {
    queueMicrotask(resolve);
  });

const configure = (overrides: Partial<LoggingConfig> = {}): void =>
  configureLogging({ ...DEFAULT_LOGGING_CONFIG, level: 'trace', ...overrides });

beforeEach(() => {
  resetLogging();
  configure();
  vi.setSystemTime(new Date('2026-09-22T00:00:00.000Z'));
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('recording filters', () => {
  it('records warn and above across every namespace by default', () => {
    resetLogging();

    expect(getLoggingConfig()).toEqual(DEFAULT_LOGGING_CONFIG);
    createLogger({ area: 'boot', namespace: 'app' }).info('quiet');
    createLogger({ area: 'boot', namespace: 'app' }).warn('loud');
    createLogger({ area: 'save', namespace: 'persistence' }).error('failed');

    expect(getLogSnapshot().entries.map((entry) => entry.message)).toEqual(['failed', 'loud']);
    expect(getLoggingConfig().namespaces).toEqual(LOG_NAMESPACES);
  });

  it('applies severity and namespace selections', () => {
    configure({ level: 'warn', namespaces: ['queue'] });

    createLogger({ area: 'runtime', namespace: 'workflows' }).error('wrong namespace');
    createLogger({ area: 'runtime', namespace: 'queue' }).debug('too quiet');
    createLogger({ area: 'runtime', namespace: 'queue' }).warn('recorded');

    expect(getLogSnapshot().entries).toMatchObject([{ level: 'warn', message: 'recorded' }]);
  });
});

describe('entries', () => {
  it('stores immutable normalized snapshots with error detail and source metadata', () => {
    const context = { nested: { token: 'secret', value: 1 } };
    const logger = createLogger({ area: 'compile', namespace: 'workflows', projectId: 'project-a' }).child({
      operationId: 'op-1',
      widget: { instanceId: 'workflow:center', region: 'center', typeId: 'workflow' },
    });

    logger.error({ context, error: new TypeError('bad graph'), message: 'Compile failed', name: 'workflow.compile' });
    context.nested.value = 2;

    const [entry] = getLogSnapshot().entries;

    expect(entry).toMatchObject({
      context: { nested: { token: '[redacted]', value: 1 } },
      createdAt: '2026-09-22T00:00:00.000Z',
      error: { message: 'bad graph', name: 'TypeError' },
      id: expect.stringMatching(/^log-\d+$/),
      level: 'error',
      message: 'Compile failed',
      name: 'workflow.compile',
      namespace: 'workflows',
      sequence: expect.any(Number),
      source: {
        area: 'compile',
        namespace: 'workflows',
        operationId: 'op-1',
        projectId: 'project-a',
        widget: { instanceId: 'workflow:center', region: 'center', typeId: 'workflow' },
      },
    });
    expect(Object.isFrozen(entry)).toBe(true);
    expect(Object.isFrozen(entry?.context)).toBe(true);
    expect(Object.isFrozen(entry?.source)).toBe(true);
    expect(entry?.error?.stack).toContain('TypeError');
  });

  it('bounds messages and strips bare URL queries like any other string', () => {
    const logger = createLogger({ area: 'runtime', namespace: 'system' });

    logger.warn('https://host.example/api?token=abc');
    logger.warn(`Request failed for ${'m'.repeat(1_200)}`);

    const [long, url] = getLogSnapshot().entries;

    expect(url?.message).toBe('https://host.example/api');
    expect(long?.message.length).toBeLessThan(1_100);
    expect(long?.truncated).toBe(true);
  });

  it('caps one entry at 8 KiB and marks the truncation', () => {
    const logger = createLogger({ area: 'runtime', namespace: 'system' });

    logger.error({ context: { rows: Array.from({ length: 40 }, () => 'y'.repeat(900)) }, message: 'huge' });

    const [entry] = getLogSnapshot().entries;

    expect(entry?.truncated).toBe(true);
    expect(typeof entry?.context).toBe('string');
    expect(entry?.context).toMatch(/^\[context omitted: \d+ bytes\]$/);
    expect(JSON.stringify(entry).length).toBeLessThanOrEqual(8 * 1024);
    expect(getLogSnapshot().truncatedCount).toBe(1);
  });
});

describe('retention', () => {
  it('keeps problems, verbose events and timings in separate bounded buffers', () => {
    configure({ performanceTimingsEnabled: true });
    const logger = createLogger({ area: 'runtime', namespace: 'system' });

    for (let index = 0; index < LOG_RETENTION_LIMITS.verbose + 5; index += 1) {
      logger.debug(`debug ${index}`);
    }
    logger.error('problem');
    for (let index = 0; index < LOG_RETENTION_LIMITS.timings + 2; index += 1) {
      recordTiming({ area: 'runtime', namespace: 'workflows' }, `timing ${index}`, index);
    }

    const { entries, retention } = getLogSnapshot();

    expect(retention).toEqual({
      problems: { count: 1, evicted: 0, limit: LOG_RETENTION_LIMITS.problems },
      timings: { count: LOG_RETENTION_LIMITS.timings, evicted: 2, limit: LOG_RETENTION_LIMITS.timings },
      verbose: { count: LOG_RETENTION_LIMITS.verbose, evicted: 5, limit: LOG_RETENTION_LIMITS.verbose },
    });
    expect(entries.find((entry) => entry.message === 'problem')).toBeDefined();
    expect(entries.find((entry) => entry.message === 'debug 0')).toBeUndefined();
    expect(entries.find((entry) => entry.message === 'debug 5')).toBeDefined();
  });
});

describe('publication', () => {
  it('returns a stable snapshot until the next change and coalesces listeners per microtask', async () => {
    const listener = vi.fn();
    const unsubscribe = subscribeLogs(listener);
    const logger = createLogger({ area: 'runtime', namespace: 'system' });
    const before = getLogSnapshot();

    logger.warn('a');
    logger.warn('b');
    logger.warn('c');

    expect(listener).not.toHaveBeenCalled();
    expect(getLogSnapshot().entries).toHaveLength(3);
    expect(getLogSnapshot()).toBe(getLogSnapshot());
    expect(getLogSnapshot()).not.toBe(before);

    await flushMicrotasks();

    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
  });

  it('clears one project scope without touching other projects or application events', () => {
    createLogger({ area: 'boot', namespace: 'app' }).warn('application');
    createLogger({ area: 'save', namespace: 'persistence', projectId: 'project-a' }).warn('a');
    createLogger({ area: 'save', namespace: 'persistence', projectId: 'project-b' }).warn('b');

    clearLogs({ kind: 'project', projectId: 'project-a' });
    expect(getLogSnapshot().entries.map((entry) => entry.message)).toEqual(['b', 'application']);

    clearLogs({ kind: 'project-and-application', projectId: 'project-b' });
    expect(getLogSnapshot().entries).toEqual([]);
  });
});

describe('account lifetimes', () => {
  it('clears history and restores defaults synchronously when the account changes', () => {
    const listener = vi.fn();
    const unsubscribe = subscribeLogs(listener);

    accountLifecycle.activate('user-a');
    configure({ level: 'trace' });
    listener.mockClear();
    createLogger({ area: 'runtime', namespace: 'system' }).debug('user a');

    accountLifecycle.invalidate();

    expect(listener).toHaveBeenCalledTimes(1);
    expect(getLogSnapshot().entries).toEqual([]);
    expect(getLoggingConfig()).toEqual(DEFAULT_LOGGING_CONFIG);
    unsubscribe();
  });

  it('rejects writes from loggers whose owning lifetime expired', async () => {
    accountLifecycle.activate('user-a');
    const owner = captureAccountScope();
    const logger = createLogger({ area: 'runtime', namespace: 'queue' }, { owner });
    const listener = vi.fn();
    const unsubscribe = subscribeLogs(listener);

    logger.error('accepted');
    accountLifecycle.invalidate();
    configure();
    listener.mockClear();

    logger.error('late');
    recordTiming({ area: 'runtime', namespace: 'queue' }, 'late', 1, owner);
    await flushMicrotasks();

    expect(getLogSnapshot().entries).toEqual([]);
    expect(listener).not.toHaveBeenCalled();

    createLogger({ area: 'runtime', namespace: 'queue' }, { owner: captureAccountScope() }).error('current');
    expect(getLogSnapshot().entries).toHaveLength(1);
    unsubscribe();
  });
});
