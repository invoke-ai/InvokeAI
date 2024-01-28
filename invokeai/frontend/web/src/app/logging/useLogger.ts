import { createLogWriter } from '@roarr/browser-log-writer';
import { useAppSelector } from 'app/store/storeHooks';
import { useEffect, useMemo } from 'react';
import { ROARR, Roarr } from 'roarr';

import type { LoggerNamespace } from './logger';
import { $logger, BASE_CONTEXT, LOG_LEVEL_MAP, logger } from './logger';

export const useLogger = (namespace: LoggerNamespace) => {
  const consoleLogLevel = useAppSelector((s) => s.system.consoleLogLevel);
  const shouldLogToConsole = useAppSelector((s) => s.system.shouldLogToConsole);

  // The provided Roarr browser log writer uses localStorage to config logging to console
  useEffect(() => {
    if (shouldLogToConsole) {
      // Enable console log output
      localStorage.setItem('ROARR_LOG', 'true');

      // Use a filter to show only logs of the given level
      localStorage.setItem('ROARR_FILTER', `context.logLevel:>=${LOG_LEVEL_MAP[consoleLogLevel]}`);
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
