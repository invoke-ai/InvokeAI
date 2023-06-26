export const mode =
  (light: string, dark: string) => (colorMode: 'light' | 'dark') =>
    colorMode === 'light' ? light : dark;
