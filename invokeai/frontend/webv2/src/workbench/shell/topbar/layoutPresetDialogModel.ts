export const getNextIconId = (iconIds: readonly string[], currentIconId: string, key: string): string | null => {
  if (iconIds.length === 0) {
    return null;
  }
  if (key === 'Home') {
    return iconIds[0] ?? null;
  }
  if (key === 'End') {
    return iconIds.at(-1) ?? null;
  }

  const direction = key === 'ArrowRight' || key === 'ArrowDown' ? 1 : key === 'ArrowLeft' || key === 'ArrowUp' ? -1 : 0;

  if (direction === 0) {
    return null;
  }

  const currentIndex = iconIds.indexOf(currentIconId);
  const nextIndex = (Math.max(currentIndex, 0) + direction + iconIds.length) % iconIds.length;

  return iconIds[nextIndex] ?? null;
};
