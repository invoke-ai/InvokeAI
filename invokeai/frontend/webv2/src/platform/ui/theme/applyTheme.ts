import { DEFAULT_THEME, THEMES_BY_ID } from './themes';

/** Apply root theme attributes without persisting hints; transient previews must not overwrite saved appearance. */
export const applyThemeToRoot = (themeId: string): void => {
  const root = document.documentElement;
  const theme = THEMES_BY_ID[themeId as keyof typeof THEMES_BY_ID] ?? DEFAULT_THEME;

  root.dataset.theme = theme.id;
  root.style.colorScheme = theme.colorScheme;
  // Keep native color-mode classes aligned for browser chrome and unoverridden Chakra defaults.
  root.classList.toggle('dark', theme.colorScheme === 'dark');
  root.classList.toggle('light', theme.colorScheme === 'light');
};
