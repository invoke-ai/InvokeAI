import { DEFAULT_THEME, THEMES_BY_ID } from './themes';

/**
 * Applies a theme to the document root. Semantic tokens key off
 * `:root[data-theme]`, so these attribute writes restyle the whole shell with
 * no React re-render. Deliberately does NOT touch the pre-paint localStorage
 * hint — callers that persist (ThemeController) own that; transient callers
 * (palette live-preview) must not.
 */
export const applyThemeToRoot = (themeId: string): void => {
  const root = document.documentElement;
  const theme = THEMES_BY_ID[themeId as keyof typeof THEMES_BY_ID] ?? DEFAULT_THEME;

  root.dataset.theme = theme.id;
  root.style.colorScheme = theme.colorScheme;
  // Keep the native color-mode class in sync for unthemed browser chrome
  // (scrollbars, native pickers) and any Chakra defaults we don't override.
  root.classList.toggle('dark', theme.colorScheme === 'dark');
  root.classList.toggle('light', theme.colorScheme === 'light');
};
