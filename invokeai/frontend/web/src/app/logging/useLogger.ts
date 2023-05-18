import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';
import { isEqual } from 'lodash-es';
import { useEffect } from 'react';
import { LogLevelName, ROARR, Roarr } from 'roarr';
import { createLogWriter } from '@roarr/browser-log-writer';

// Base logging context includes only the package name
const baseContext = { package: '@invoke-ai/invoke-ai-ui' };

// Create browser log writer
ROARR.write = createLogWriter();

// Module-scoped logger - can be imported and used anywhere
export let log = Roarr.child(baseContext);

// Translate human-readable log levels to numbers, used for log filtering
export const LOG_LEVEL_MAP: Record<LogLevelName, number> = {
  trace: 10,
  debug: 20,
  info: 30,
  warn: 40,
  error: 50,
  fatal: 60,
};

export const VALID_LOG_LEVELS = [
  'trace',
  'debug',
  'info',
  'warn',
  'error',
  'fatal',
] as const;

export type InvokeLogLevel = (typeof VALID_LOG_LEVELS)[number];

const selector = createSelector(
  systemSelector,
  (system) => {
    const { app_version, consoleLogLevel, shouldLogToConsole } = system;

    return {
      version: app_version,
      consoleLogLevel,
      shouldLogToConsole,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export const useLogger = () => {
  const { version, consoleLogLevel, shouldLogToConsole } =
    useAppSelector(selector);

  // The provided Roarr browser log writer uses localStorage to config logging to console
  useEffect(() => {
    if (shouldLogToConsole) {
      // Enable console log output
      localStorage.setItem('ROARR_LOG', 'true');

      // Use a filter to show only logs of the given level
      localStorage.setItem(
        'ROARR_FILTER',
        `context.logLevel:>=${LOG_LEVEL_MAP[consoleLogLevel]}`
      );
    } else {
      // Disable console log output
      localStorage.setItem('ROARR_LOG', 'false');
    }
    ROARR.write = createLogWriter();
  }, [consoleLogLevel, shouldLogToConsole]);

  // Update the module-scoped logger context as needed
  useEffect(() => {
    const newContext: Record<string, any> = {
      ...baseContext,
    };

    if (version) {
      newContext.version = version;
    }

    log = Roarr.child(newContext);
  }, [version]);

  // Use the logger within components - no different than just importing it directly
  return log;
};
