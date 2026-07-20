import type { WorkflowReadPort } from '@features/workflow/react';
import type { EqualityFn } from '@platform/state/selectors';

/** Maps a broad external store to a referentially stable narrow read port. */
export const createCachedWorkflowReadPort = <Source, Snapshot>(
  subscribe: (listener: () => void) => () => void,
  getSourceSnapshot: () => Source,
  mapSnapshot: (source: Source) => Snapshot,
  isEqual: EqualityFn<Snapshot> = Object.is
): WorkflowReadPort<Snapshot> => {
  let sourceSnapshot = getSourceSnapshot();
  let snapshot = mapSnapshot(sourceSnapshot);

  const refresh = (): boolean => {
    const nextSourceSnapshot = getSourceSnapshot();
    if (Object.is(nextSourceSnapshot, sourceSnapshot)) {
      return false;
    }
    sourceSnapshot = nextSourceSnapshot;
    const nextSnapshot = mapSnapshot(nextSourceSnapshot);
    if (isEqual(snapshot, nextSnapshot)) {
      return false;
    }
    snapshot = nextSnapshot;
    return true;
  };

  return {
    getSnapshot: () => {
      refresh();
      return snapshot;
    },
    subscribe: (listener) =>
      subscribe(() => {
        if (refresh()) {
          listener();
        }
      }),
  };
};
