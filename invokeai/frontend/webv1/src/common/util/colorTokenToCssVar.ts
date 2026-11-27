export const colorTokenToCssVar = (colorToken: string) => `var(--invoke-colors-${colorToken.split('.').join('-')})`;
