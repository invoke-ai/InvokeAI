import type { Dispatch } from 'react';

import type { Project, WorkbenchState } from './types';

import { createExternalStore } from './externalStore';
import { createInitialWorkbenchState, workbenchReducer, type WorkbenchAction } from './workbenchState';

export interface WorkbenchSnapshot {
  state: WorkbenchState;
  activeProject: Project;
  hasHydrated: boolean;
}

export interface WorkbenchStore {
  dispatch: Dispatch<WorkbenchAction>;
  getPersistedRevision: () => number;
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

const hasPersistedStateChanged = (previous: WorkbenchState, next: WorkbenchState): boolean =>
  !Object.is(previous.account, next.account) ||
  previous.activeProjectId !== next.activeProjectId ||
  !Object.is(previous.errorLog, next.errorLog) ||
  !Object.is(previous.projects, next.projects) ||
  !Object.is(previous.widgetFailures, next.widgetFailures);

export const createWorkbenchStore = (initialState = createInitialWorkbenchState()): WorkbenchStore => {
  let state = initialState;
  let hasHydrated = false;
  let persistedRevision = 0;
  const snapshotStore = createExternalStore(createSnapshot(state, hasHydrated));

  const setSnapshotState = (nextState: WorkbenchState, nextHasHydrated = hasHydrated): void => {
    if (Object.is(nextState, state) && nextHasHydrated === hasHydrated) {
      return;
    }

    if (!Object.is(nextState, state) && hasPersistedStateChanged(state, nextState)) {
      persistedRevision += 1;
    }

    state = nextState;
    hasHydrated = nextHasHydrated;
    snapshotStore.setSnapshot(createSnapshot(state, hasHydrated));
  };

  return {
    dispatch(action) {
      setSnapshotState(workbenchReducer(state, action));
    },
    getPersistedRevision: () => persistedRevision,
    getSnapshot: snapshotStore.getSnapshot,
    getState: () => state,
    setHasHydrated: (nextHasHydrated) => setSnapshotState(state, nextHasHydrated),
    subscribe: snapshotStore.subscribe,
  };
};

export type { WorkbenchAction };
