import { useMemo } from 'react';

import type { LogNamespace } from './logger';
import { logger } from './logger';

export const useLogger = (namespace: LogNamespace) => {
  const log = useMemo(() => logger(namespace), [namespace]);

  return log;
};
