import { useSyncExternalStore } from 'react';

import type { LoggingConfig, LogSnapshot } from './contracts';

import { getLoggingConfig, getLogSnapshot, subscribeLogs } from './logger';

export const useLogSnapshot = (): LogSnapshot => useSyncExternalStore(subscribeLogs, getLogSnapshot, getLogSnapshot);

export const useLoggingConfig = (): LoggingConfig =>
  useSyncExternalStore(subscribeLogs, getLoggingConfig, getLoggingConfig);
