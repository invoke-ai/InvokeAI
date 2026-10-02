import { shallowEqual } from '@platform/state/selectors';
import { applyThemeToRoot } from '@theme/applyTheme';
import { DEFAULT_THEME, THEMES_BY_ID } from '@theme/system';
import { useWorkbenchSettingsSelector } from '@workbench/settings/store';
import { useLayoutEffect } from 'react';

/** Root data attributes drive theme and motion CSS without React rerenders. */
/** Pre-paint hints are account-independent because identity is unknown at first paint. */
const THEME_HINT_STORAGE_KEY = 'invokeai:v7:webv2:theme';
const REDUCE_MOTION_HINT_STORAGE_KEY = 'invokeai:v7:webv2:reduce-motion';
const HIGH_CONTRAST_HINT_STORAGE_KEY = 'invokeai:v7:webv2:high-contrast';

/** Mirrors a boolean appearance flag onto `<html data-*>` and its pre-paint hint. */
const applyRootFlag = (dataKey: 'highContrast' | 'reduceMotion', hintKey: string, enabled: boolean): void => {
  const root = document.documentElement;
  if (enabled) {
    root.dataset[dataKey] = 'true';
  } else {
    delete root.dataset[dataKey];
  }
  try {
    if (enabled) {
      window.localStorage.setItem(hintKey, 'true');
    } else {
      window.localStorage.removeItem(hintKey);
    }
  } catch {
    // Storage unavailable — the next load just waits for settings to resolve.
  }
};

export const ThemeController = () => {
  const { highContrast, reduceMotion, status, themeId } = useWorkbenchSettingsSelector(
    (snapshot) => ({
      highContrast: snapshot.preferences.highContrast,
      reduceMotion: snapshot.preferences.reduceMotion,
      status: snapshot.status,
      themeId: snapshot.preferences.themeId,
    }),
    shallowEqual
  );
  // Keep pre-paint hints until settings resolve; applying defaults would flash the theme.
  const hasResolved = status === 'ready' || status === 'error';

  useLayoutEffect(() => {
    if (!hasResolved) {
      return;
    }

    const theme = THEMES_BY_ID[themeId] ?? DEFAULT_THEME;

    applyThemeToRoot(theme.id);

    try {
      window.localStorage.setItem(THEME_HINT_STORAGE_KEY, theme.id);
    } catch {
      // Storage unavailable — the next load just paints the default theme.
    }
  }, [hasResolved, themeId]);

  useLayoutEffect(() => {
    if (hasResolved) {
      applyRootFlag('reduceMotion', REDUCE_MOTION_HINT_STORAGE_KEY, reduceMotion);
    }
  }, [hasResolved, reduceMotion]);

  useLayoutEffect(() => {
    if (hasResolved) {
      applyRootFlag('highContrast', HIGH_CONTRAST_HINT_STORAGE_KEY, highContrast);
    }
  }, [hasResolved, highContrast]);

  return null;
};
