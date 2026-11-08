import type { LoggingConfig } from '@platform/logging/contracts';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';

import { configureLogging, createLogger, getLoggingConfig } from '@platform/logging/logger';
import { APP_VERSION } from '@platform/runtime/appMetadata';
import { toLoggingConfig } from '@workbench/diagnostics/loggingPreferences';
import { subscribeWorkbenchPreferences } from '@workbench/settings/store';

const appLogger = createLogger({ area: 'browser', namespace: 'app' });

interface GlobalErrorTarget {
  addEventListener(type: 'error', listener: (event: ErrorEvent) => void): void;
  addEventListener(type: 'unhandledrejection', listener: (event: PromiseRejectionEvent) => void): void;
}

/**
 * Record uncaught errors and rejections as application events without suppressing the browser's own reporting.
 * They are not attributed to whichever project happens to be active.
 */
export const installGlobalErrorLogging = (target: GlobalErrorTarget): void => {
  target.addEventListener('error', (event) => {
    appLogger.error({
      context: { column: event.colno, file: event.filename, line: event.lineno },
      error: event.error ?? event.message,
      message: event.message || 'Uncaught error',
      name: 'app.uncaught-error',
    });
  });
  target.addEventListener('unhandledrejection', (event) => {
    appLogger.error({
      error: event.reason,
      message: 'Unhandled promise rejection',
      name: 'app.unhandled-rejection',
    });
  });
};

const isSameConfig = (left: LoggingConfig, right: LoggingConfig): boolean =>
  (Object.keys(left) as (keyof LoggingConfig)[]).every((key) => {
    const a = left[key];
    const b = right[key];

    return Array.isArray(a) && Array.isArray(b)
      ? a.length === b.length && a.every((item, index) => item === b[index])
      : a === b;
  });

/** Apply saved preferences to the recorder; publishes only when the effective configuration changes. */
export const applyLoggingPreferences = (preferences: WorkbenchPreferences): void => {
  const next = toLoggingConfig(preferences);

  if (!isSameConfig(getLoggingConfig(), next)) {
    configureLogging(next);
  }
};

let isConfigured = false;

/** App keeps capture settings in step with saved preferences for whichever account is active. */
export const configureAppLogging = (): void => {
  if (isConfigured) {
    return;
  }
  isConfigured = true;

  subscribeWorkbenchPreferences(applyLoggingPreferences);
  installGlobalErrorLogging(window);
  createLogger({ area: 'boot', namespace: 'app' }).info({
    context: { version: APP_VERSION },
    message: 'Application starting',
    name: 'app.boot',
  });
};
