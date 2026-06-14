import type { Dispatch } from 'react';

import type { Project, WorkbenchState } from './types';
import { createInitialWorkbenchState, workbenchReducer, type WorkbenchAction } from './workbenchState';

export interface WorkbenchSnapshot {
  state: WorkbenchState;
  activeProject: Project;
  hasHydrated: boolean;
}

export interface WorkbenchStore {
  dispatch: Dispatch<WorkbenchAction>;
  getSnapshot: () => WorkbenchSnapshot;
  getState: () => WorkbenchState;
  setHasHydrated: (hasHydrated: boolean) => void;
  subscribe: (listener: () => void) => () => void;
}

const getActiveProject = (state: WorkbenchState): Project =>
  state.projects.find((project) => project.id === state.activeProjectId) ?? state.projects[0];

const createSnapshot = (state: WorkbenchState, hasHydrated: boolean): WorkbenchSnapshot => ({
  activeProject: getActiveProject(state),
  hasHydrated,
  state,
});

export const createWorkbenchStore = (initialState = createInitialWorkbenchState()): WorkbenchStore => {
  let state = initialState;
  let hasHydrated = false;
  let snapshot = createSnapshot(state, hasHydrated);
  const listeners = new Set<() => void>();

  const emit = (): void => {
    for (const listener of listeners) {
      listener();
    }
  };

  const setSnapshotState = (nextState: WorkbenchState, nextHasHydrated = hasHydrated): void => {
    if (Object.is(nextState, state) && nextHasHydrated === hasHydrated) {
      return;
    }

    state = nextState;
    hasHydrated = nextHasHydrated;
    snapshot = createSnapshot(state, hasHydrated);
    emit();
  };

  return {
    dispatch(action) {
      setSnapshotState(workbenchReducer(state, action));
    },
    getSnapshot: () => snapshot,
    getState: () => state,
    setHasHydrated: (nextHasHydrated) => setSnapshotState(state, nextHasHydrated),
    subscribe(listener) {
      listeners.add(listener);

      return () => {
        listeners.delete(listener);
      };
    },
  };
};

export type { WorkbenchAction };
