/**
 * Returns the parent directory of a filesystem path for display. Handles both POSIX and
 * Windows separators because the backend returns platform-native paths.
 *
 * Returns null if the path contains no separator (so callers can hide the label entirely
 * rather than rendering an empty string).
 */
export const getParentDirectory = (path: string): string | null => {
  const lastSlash = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
  if (lastSlash <= 0) {
    return null;
  }
  return path.substring(0, lastSlash);
};
