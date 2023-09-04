export const colorTokenToCssVar = (colorToken: string) =>
  `var(--invokeai-colors-${colorToken.split('.').join('-')})`;
