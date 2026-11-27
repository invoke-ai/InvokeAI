/**
 * One reading of the project's queue and the connection, shared by the top-bar queue button, the status-bar chip and
 * the screen-reader announcement, so none of them reads as idle while open work waits on a lost connection.
 */
export type QueueActivity = 'idle' | 'loading-model' | 'offline' | 'paused' | 'running';

export const getQueueActivity = ({
  hasOpenWork,
  isConnected,
  isLoadingModels,
  isPaused,
}: {
  hasOpenWork: boolean;
  isConnected: boolean;
  isLoadingModels: boolean;
  isPaused: boolean;
}): QueueActivity => {
  if (!hasOpenWork) {
    return 'idle';
  }

  // Offline first: the processor's paused state is a stale reading until the connection returns.
  if (!isConnected) {
    return 'offline';
  }

  if (isPaused) {
    return 'paused';
  }

  return isLoadingModels ? 'loading-model' : 'running';
};
