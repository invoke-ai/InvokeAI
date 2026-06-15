import { useEffect } from 'react';

import { DEFAULT_THEME, THEMES_BY_ID } from '@theme/system';
import { useWorkbenchSettings } from './settings/store';

/**
 * Applies the persisted appearance preferences to the document root.
 *
 * Theme switching is intentionally a DOM-attribute flip rather than a React
 * re-theme: the semantic-token conditions in `theme/system.ts` key off
 * `<html data-theme>`, so changing the attribute restyles the whole shell with
 * no component re-render. `data-reduce-motion` is read by the global CSS that
 * neutralizes transitions. Renders nothing.
 */
/**
 * Read by the pre-paint script in index.html. A dedicated key (rather than the
 * workbench snapshot, which is per-user on multi-user backends) so the first
 * paint can apply the last-used theme without knowing who is signed in.
 */
const THEME_HINT_STORAGE_KEY = 'invokeai:v7:webv2:theme';

export const ThemeController = () => {
  const { preferences, status } = useWorkbenchSettings();
  const { reduceMotion, themeId } = preferences;
  // Until the settings store has resolved, the pre-paint hint script owns the
  // theme; applying the store's defaults here would flash and clobber it.
  const hasResolved = status === 'ready' || status === 'error';

  useEffect(() => {
    if (!hasResolved) {
      return;
    }

    const root = document.documentElement;
    const theme = THEMES_BY_ID[themeId] ?? DEFAULT_THEME;

    root.dataset.theme = theme.id;
    root.style.colorScheme = theme.colorScheme;
    // Keep the native color-mode class in sync for unthemed browser chrome
    // (scrollbars, native pickers) and any Chakra defaults we don't override.
    root.classList.toggle('dark', theme.colorScheme === 'dark');
    root.classList.toggle('light', theme.colorScheme === 'light');

    try {
      window.localStorage.setItem(THEME_HINT_STORAGE_KEY, theme.id);
    } catch {
      // Storage unavailable — the next load just paints the default theme.
    }
  }, [hasResolved, themeId]);

  useEffect(() => {
    if (!hasResolved) {
      return;
    }

    const root = document.documentElement;

    if (reduceMotion) {
      root.dataset.reduceMotion = 'true';
    } else {
      delete root.dataset.reduceMotion;
    }
  }, [hasResolved, reduceMotion]);

  return null;
};
