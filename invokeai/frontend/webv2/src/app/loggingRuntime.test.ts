import { DEFAULT_LOGGING_CONFIG } from '@platform/logging/contracts';
import {
  configureLogging,
  getLoggingConfig,
  getLogSnapshot,
  resetLogging,
  subscribeLogs,
} from '@platform/logging/logger';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { normalizeWorkbenchPreferences } from '@workbench/settings/store';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { applyLoggingPreferences, installGlobalErrorLogging } from './loggingRuntime';

type Listener = (event: never) => void;

const createTarget = () => {
  const listeners = new Map<string, Listener[]>();

  return {
    addEventListener: (type: string, listener: Listener) => {
      listeners.set(type, [...(listeners.get(type) ?? []), listener]);
    },
    dispatch: (type: string, event: unknown) => {
      for (const listener of listeners.get(type) ?? []) {
        listener(event as never);
      }
    },
  };
};

beforeEach(() => {
  resetLogging();
  configureLogging({ ...DEFAULT_LOGGING_CONFIG, level: 'debug' });
});

describe('applyLoggingPreferences', () => {
  it('applies every preference field and publishes only on effective changes', () => {
    const listener = vi.fn();
    const unsubscribe = subscribeLogs(listener);
    const preferences = normalizeWorkbenchPreferences({
      developerConsoleOutputEnabled: true,
      developerLogEnabled: false,
      developerLogLevel: 'info',
      developerLogNamespaces: ['queue', 'canvas'],
      developerPerformanceTimingsEnabled: true,
    });

    applyLoggingPreferences(preferences);
    applyLoggingPreferences({ ...preferences, developerLogNamespaces: ['canvas', 'queue'] });

    expect(getLoggingConfig()).toEqual({
      consoleOutputEnabled: true,
      enabled: false,
      level: 'info',
      namespaces: ['canvas', 'queue'],
      performanceTimingsEnabled: true,
    });
    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
  });

  it('ends in the hydrated account preferences whichever order cleanup runs', () => {
    accountLifecycle.activate('user-a');
    applyLoggingPreferences(normalizeWorkbenchPreferences({ developerLogLevel: 'trace' }));

    accountLifecycle.activate('user-b');
    expect(getLoggingConfig().level).toBe('warn');

    applyLoggingPreferences(normalizeWorkbenchPreferences({ developerLogLevel: 'error' }));
    expect(getLoggingConfig().level).toBe('error');
  });
});

describe('installGlobalErrorLogging', () => {
  it('records uncaught errors and rejections as unattributed application events', () => {
    const target = createTarget();

    installGlobalErrorLogging(target);
    target.dispatch('error', {
      colno: 7,
      error: new RangeError('boom'),
      filename: 'https://host.example/assets/app.js?v=1',
      lineno: 12,
      message: 'Uncaught RangeError: boom',
    });
    target.dispatch('unhandledrejection', { reason: 'rejected value' });

    expect(getLogSnapshot().entries).toMatchObject([
      {
        error: { message: 'rejected value', name: 'NonError' },
        level: 'error',
        name: 'app.unhandled-rejection',
        source: { area: 'browser', namespace: 'app' },
      },
      {
        context: { column: 7, file: 'https://host.example/assets/app.js', line: 12 },
        error: { message: 'boom', name: 'RangeError' },
        level: 'error',
        message: 'Uncaught RangeError: boom',
        name: 'app.uncaught-error',
      },
    ]);
    expect(getLogSnapshot().entries[1]?.source).not.toHaveProperty('projectId');
  });
});
