import type { LoggingConfig } from '@platform/logging/contracts';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';

import { DEFAULT_PREFERENCES, patchWorkbenchPreferences } from '@workbench/settings/store';

export const LOGGING_PREFERENCE_KEYS = [
  'developerConsoleOutputEnabled',
  'developerLogEnabled',
  'developerLogLevel',
  'developerLogNamespaces',
  'developerPerformanceTimingsEnabled',
] as const satisfies readonly (keyof WorkbenchPreferences)[];

export const toLoggingConfig = (preferences: WorkbenchPreferences): LoggingConfig => ({
  consoleOutputEnabled: preferences.developerConsoleOutputEnabled,
  enabled: preferences.developerLogEnabled,
  level: preferences.developerLogLevel,
  namespaces: preferences.developerLogNamespaces,
  performanceTimingsEnabled: preferences.developerPerformanceTimingsEnabled,
});

export const areLoggingPreferencesDefault = (preferences: WorkbenchPreferences): boolean =>
  LOGGING_PREFERENCE_KEYS.every((key) => {
    const value = preferences[key];
    const fallback = DEFAULT_PREFERENCES[key];

    return Array.isArray(value) && Array.isArray(fallback)
      ? value.length === fallback.length && value.every((item, index) => item === fallback[index])
      : value === fallback;
  });

/** Restores default capture settings, including namespaces saved before newer namespaces existed. */
export const resetLoggingPreferences = (): Promise<void> =>
  patchWorkbenchPreferences({
    developerConsoleOutputEnabled: DEFAULT_PREFERENCES.developerConsoleOutputEnabled,
    developerLogEnabled: DEFAULT_PREFERENCES.developerLogEnabled,
    developerLogLevel: DEFAULT_PREFERENCES.developerLogLevel,
    developerLogNamespaces: [...DEFAULT_PREFERENCES.developerLogNamespaces],
    developerPerformanceTimingsEnabled: DEFAULT_PREFERENCES.developerPerformanceTimingsEnabled,
  });
