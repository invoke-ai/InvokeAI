export interface SingleFlight<T> {
  /** Runs task unless a run with the same key is in flight; concurrent callers share the promise. */
  run(key: string, task: () => Promise<T>): Promise<T>;
}

export const createSingleFlight = <T>(): SingleFlight<T> => {
  let pending: Promise<T> | null = null;
  let pendingKey: string | null = null;

  return {
    run(key, task) {
      if (pending && pendingKey === key) {
        return pending;
      }
      const flight = task().finally(() => {
        // A newer flight for a different key may have replaced this one; only clear our own.
        if (pending === flight) {
          pending = null;
          pendingKey = null;
        }
      });
      pending = flight;
      pendingKey = key;
      return flight;
    },
  };
};
