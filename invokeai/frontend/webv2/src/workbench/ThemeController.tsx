import { useEffect } from 'react';

import { THEMES_BY_ID } from '../theme/system';
import { useWorkbench } from './WorkbenchContext';

/**
 * Applies the persisted appearance preferences to the document root.
 *
 * Theme switching is intentionally a DOM-attribute flip rather than a React
 * re-theme: the semantic-token conditions in `theme/system.ts` key off
 * `<html data-theme>`, so changing the attribute restyles the whole shell with
 * no component re-render. `data-reduce-motion` is read by the global CSS that
 * neutralizes transitions. Renders nothing.
 */
export const ThemeController = () => {
  const { state } = useWorkbench();
  const { reduceMotion, themeId } = state.account.preferences;

  useEffect(() => {
    const root = document.documentElement;
    const theme = THEMES_BY_ID[themeId] ?? THEMES_BY_ID.dark;

    root.dataset.theme = theme.id;
    root.style.colorScheme = theme.colorScheme;
    // Keep the native color-mode class in sync for unthemed browser chrome
    // (scrollbars, native pickers) and any Chakra defaults we don't override.
    root.classList.toggle('dark', theme.colorScheme === 'dark');
    root.classList.toggle('light', theme.colorScheme === 'light');
  }, [themeId]);

  useEffect(() => {
    const root = document.documentElement;

    if (reduceMotion) {
      root.dataset.reduceMotion = 'true';
    } else {
      delete root.dataset.reduceMotion;
    }
  }, [reduceMotion]);

  return null;
};
