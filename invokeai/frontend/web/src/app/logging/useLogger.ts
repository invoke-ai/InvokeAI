import { createLogWriter } from '@roarr/browser-log-writer';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectSystemLogIsEnabled,
  selectSystemLogLevel,
  selectSystemLogNamespaces,
} from 'features/system/store/systemSlice';
import { useEffect, useMemo } from 'react';
import { ROARR, Roarr } from 'roarr';

import type { LogNamespace } from './logger';
import { $logger, BASE_CONTEXT, LOG_LEVEL_MAP, logger } from './logger';

export const useLogger = (namespace: LogNamespace) => {
  const logLevel = useAppSelector(selectSystemLogLevel);
  const logNamespaces = useAppSelector(selectSystemLogNamespaces);
  const logIsEnabled = useAppSelector(selectSystemLogIsEnabled);

  // The provided Roarr browser log writer uses localStorage to config logging to console
  useEffect(() => {
    if (logIsEnabled) {
      // Enable console log output
      localStorage.setItem('ROARR_LOG', 'true');

      // Use a filter to show only logs of the given level
      let filter = `context.logLevel:>=${LOG_LEVEL_MAP[logLevel]}`;
      if (logNamespaces.length > 0) {
        filter += ` AND (${logNamespaces.map((ns) => `context.namespace:${ns}`).join(' OR ')})`;
      } else {
        filter += ' AND context.namespace:undefined';
      }
      localStorage.setItem('ROARR_FILTER', filter);
    } else {
      // Disable console log output
      localStorage.setItem('ROARR_LOG', 'false');
    }
    ROARR.write = createLogWriter();
  }, [logLevel, logIsEnabled, logNamespaces]);

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
