/** Return no duration when either backend timestamp is missing. */
export const formatDuration = (startedAt?: string | null, completedAt?: string | null): string | null => {
  if (!startedAt || !completedAt) {
    return null;
  }

  const seconds = (new Date(completedAt).getTime() - new Date(startedAt).getTime()) / 1000;

  if (!Number.isFinite(seconds) || seconds < 0) {
    return null;
  }

  const fixed = seconds.toFixed(2);

  // Compare the rounded label, not the raw value: 59.996s would print "60.00s".
  if (Number(fixed) < 60) {
    return `${fixed}s`;
  }

  // Round to whole seconds before splitting units, so 119.6s reads "2m 0s", not "1m 60s".
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);

  if (minutes < 60) {
    return `${minutes}m ${total % 60}s`;
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
