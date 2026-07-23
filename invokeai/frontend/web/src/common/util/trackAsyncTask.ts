export const trackAsyncTask = async <T>(task: () => Promise<T>, onLoadingChanged: (isLoading: boolean) => void) => {
  onLoadingChanged(true);
  try {
    return await task();
  } finally {
    onLoadingChanged(false);
  }
};
