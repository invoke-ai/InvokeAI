import { createSelector } from '@reduxjs/toolkit';
import { createLogWriter } from '@roarr/browser-log-writer';
import { useAppSelector } from 'app/store/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';
import { isEqual } from 'lodash-es';
import { useEffect, useMemo } from 'react';
import { ROARR, Roarr } from 'roarr';
import {
  $logger,
  BASE_CONTEXT,
  LOG_LEVEL_MAP,
  LoggerNamespace,
  logger,
} from './logger';

const selector = createSelector(
  systemSelector,
  (system) => {
    const { consoleLogLevel, shouldLogToConsole } = system;

    return {
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

export const useLogger = (namespace: LoggerNamespace) => {
  const { consoleLogLevel, shouldLogToConsole } = useAppSelector(selector);

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
    // TODO: type this properly
    //eslint-disable-next-line @typescript-eslint/no-explicit-any
    const newContext: Record<string, any> = {
      ...BASE_CONTEXT,
    };

    $logger.set(Roarr.child(newContext));
  }, []);

  const log = useMemo(() => logger(namespace), [namespace]);

  return log;
};
