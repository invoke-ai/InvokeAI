export interface SingleFlight<T> {
  /** Runs task unless a run with the same key is in flight; concurrent callers share the promise. */
  run(key: string, task: () => Promise<T>): Promise<T>;
}

export interface TrailingSingleFlight {
  /**
   * Runs task now, or joins the in-flight run and queues exactly one trailing
   * rerun. Callers that join resolve when the run they joined settles; the
   * rerun keeps the result fresh but is not awaited by them.
   */
  run(task: () => Promise<void>): Promise<void>;
  /** The in-flight promise, if any (for ensure* fast paths). */
  inflight(): Promise<void> | null;
  /** Forget in-flight state and any queued rerun (account clear). */
  reset(): void;
}

/**
 * Coalesce refreshes but rerun once for changes arriving mid-flight. No caller awaits the trailing run, so its
 * rejection is swallowed.
 */
export const createTrailingSingleFlight = (): TrailingSingleFlight => {
  let pending: Promise<void> | null = null;
  let rerunRequested = false;

  const launch = (task: () => Promise<void>): Promise<void> => {
    rerunRequested = false;
    let started: Promise<void>;
    try {
      started = task();
    } catch (error) {
      started = Promise.reject(error);
    }
    const flight = started.finally(() => {
      // reset() may have run mid-flight; it cancelled any queued rerun.
      if (pending !== flight) {
        return;
      }
      pending = null;
      if (rerunRequested) {
        launch(task).catch(() => undefined);
      }
    });
    pending = flight;
    return flight;
  };

  return {
    inflight: () => pending,
    reset() {
      pending = null;
      rerunRequested = false;
    },
    run(task) {
      if (pending) {
        rerunRequested = true;
        return pending;
      }
      return launch(task);
    },
  };
};

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
