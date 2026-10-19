import { DEFAULT_LOGGING_CONFIG, LOG_NAMESPACES } from '@platform/logging/contracts';
import { DEFAULT_PREFERENCES, normalizeWorkbenchPreferences } from '@workbench/settings/store';
import { describe, expect, it } from 'vitest';

import { areLoggingPreferencesDefault, toLoggingConfig } from './loggingPreferences';

describe('loggingPreferences', () => {
  it('maps default preferences onto the default capture settings', () => {
    expect(toLoggingConfig(DEFAULT_PREFERENCES)).toEqual({
      ...DEFAULT_LOGGING_CONFIG,
      namespaces: [...LOG_NAMESPACES],
    });
    expect(areLoggingPreferencesDefault(DEFAULT_PREFERENCES)).toBe(true);
  });

  it('keeps a saved narrowed selection and console opt-in and reports them as non-default', () => {
    const preferences = normalizeWorkbenchPreferences({
      developerConsoleOutputEnabled: true,
      developerLogLevel: 'debug',
      developerLogNamespaces: ['queue'],
    });

    expect(toLoggingConfig(preferences)).toEqual({
      consoleOutputEnabled: true,
      enabled: true,
      level: 'debug',
      namespaces: ['queue'],
      performanceTimingsEnabled: false,
    });
    expect(areLoggingPreferencesDefault(preferences)).toBe(false);
  });
});
