export const isAllowedTextShortcut = (event: KeyboardEvent): boolean => {
  if (event.metaKey || event.ctrlKey) {
    const key = event.key.toLowerCase();
    return key === 'c' || key === 'v' || key === 'z' || key === 'y';
  }
  return false;
};
