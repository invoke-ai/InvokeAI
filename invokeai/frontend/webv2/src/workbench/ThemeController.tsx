import { DEFAULT_THEME, THEMES_BY_ID } from '@theme/system';
import { useLayoutEffect } from 'react';

import { useWorkbenchSettingsSelector } from './settings/store';
import { shallowEqual } from './WorkbenchContext';

/**
 * Applies the persisted appearance preferences to the document root.
 *
 * Theme switching is intentionally a DOM-attribute flip rather than a React
 * re-theme: the semantic-token conditions in `theme/system.ts` key off
 * `<html data-theme>`, so changing the attribute restyles the whole shell with
 * no component re-render. `data-reduce-motion` is read by global CSS motion
 * tokens. Renders nothing.
 */
/**
 * Read by the pre-paint script in index.html. Dedicated hint keys (rather than
 * the workbench snapshot, which is per-user on multi-user backends) let first
 * paint apply last-used appearance without knowing who is signed in.
 */
const THEME_HINT_STORAGE_KEY = 'invokeai:v7:webv2:theme';
const REDUCE_MOTION_HINT_STORAGE_KEY = 'invokeai:v7:webv2:reduce-motion';

export const ThemeController = () => {
  const { reduceMotion, status, themeId } = useWorkbenchSettingsSelector(
    (snapshot) => ({
      reduceMotion: snapshot.preferences.reduceMotion,
      status: snapshot.status,
      themeId: snapshot.preferences.themeId,
    }),
    shallowEqual
  );
  // Until the settings store has resolved, the pre-paint hint script owns the
  // theme; applying the store's defaults here would flash and clobber it.
  const hasResolved = status === 'ready' || status === 'error';

  useLayoutEffect(() => {
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

  useLayoutEffect(() => {
    if (!hasResolved) {
      return;
    }

    const root = document.documentElement;

    if (reduceMotion) {
      root.dataset.reduceMotion = 'true';
    } else {
      delete root.dataset.reduceMotion;
    }

    try {
      if (reduceMotion) {
        window.localStorage.setItem(REDUCE_MOTION_HINT_STORAGE_KEY, 'true');
      } else {
        window.localStorage.removeItem(REDUCE_MOTION_HINT_STORAGE_KEY);
      }
    } catch {
      // Storage unavailable — the next load just waits for settings to resolve.
    }
  }, [hasResolved, reduceMotion]);

  return null;
};
