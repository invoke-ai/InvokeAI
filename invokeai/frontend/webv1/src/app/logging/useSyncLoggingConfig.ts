import { configureLogging } from 'app/logging/logger';
import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import {
  selectSystemLogIsEnabled,
  selectSystemLogLevel,
  selectSystemLogNamespaces,
} from 'features/system/store/systemSlice';
import { useLayoutEffect } from 'react';

/**
 * This hook synchronizes the logging configuration stored in Redux with the logging system, which uses localstorage.
 *
 * The sync is one-way: from Redux to localstorage. This means that changes made in the UI will be reflected in the
 * logging system, but changes made directly to localstorage will not be reflected in the UI.
 *
 * See {@link configureLogging}
 */
export const useSyncLoggingConfig = () => {
  useAssertSingleton('useSyncLoggingConfig');

  const logLevel = useAppSelector(selectSystemLogLevel);
  const logNamespaces = useAppSelector(selectSystemLogNamespaces);
  const logIsEnabled = useAppSelector(selectSystemLogIsEnabled);

  useLayoutEffect(() => {
    configureLogging(logIsEnabled, logLevel, logNamespaces);
  }, [logIsEnabled, logLevel, logNamespaces]);
};
