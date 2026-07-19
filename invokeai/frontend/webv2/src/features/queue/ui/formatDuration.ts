/**
 * Compact execution-time labels for queue items ("4.37s", "1m 23s"). Computed
 * from the backend's `started_at`/`completed_at` ISO timestamps; returns null
 * when either is missing so callers can omit the field.
 */
export const formatDuration = (startedAt?: string | null, completedAt?: string | null): string | null => {
  if (!startedAt || !completedAt) {
    return null;
  }

  const seconds = (new Date(completedAt).getTime() - new Date(startedAt).getTime()) / 1000;

  if (!Number.isFinite(seconds) || seconds < 0) {
    return null;
  }

  if (seconds < 60) {
    return `${seconds.toFixed(2)}s`;
  }

  const minutes = Math.floor(seconds / 60);

  if (minutes < 60) {
    return `${minutes}m ${Math.round(seconds % 60)}s`;
  }

  const hours = Math.floor(minutes / 60);

  return `${hours}h ${minutes % 60}m`;
};

/** Ultra-compact "age" label for a timestamp ("now", "12s", "19m", "1h", "3d"). */
export const formatCompactAge = (timestamp?: string | null): string => {
  if (!timestamp) {
    return '';
  }

  const seconds = (Date.now() - new Date(timestamp).getTime()) / 1000;

  if (!Number.isFinite(seconds) || seconds < 0) {
    return '';
  }

  if (seconds < 5) {
    return 'now';
  }

  if (seconds < 60) {
    return `${Math.floor(seconds)}s`;
  }

  const minutes = Math.floor(seconds / 60);

  if (minutes < 60) {
    return `${minutes}m`;
  }

  const hours = Math.floor(minutes / 60);

  return hours < 24 ? `${hours}h` : `${Math.floor(hours / 24)}d`;
};
