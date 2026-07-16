import type { Dispatch } from 'react';

import type { Project, WorkbenchState } from './types';

import { recordDiagnosticEntry } from './diagnostics/logger';
import { createExternalStore } from './externalStore';
import { createInitialWorkbenchState, workbenchReducer, type WorkbenchReducerAction } from './workbenchState';

type WorkbenchAction = WorkbenchReducerAction;

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
  !Object.is(previous.projects, next.projects) ||
  !Object.is(previous.widgetFailures, next.widgetFailures);

const getDiagnosticProjectId = (state: WorkbenchState, projectId?: string): string | undefined =>
  projectId ?? getActiveProject(state)?.id;

const recordDiagnosticForAction = (
  action: WorkbenchAction,
  previousState: WorkbenchState,
  nextState: WorkbenchState
): void => {
  switch (action.type) {
    case 'recordError': {
      const projectId = getDiagnosticProjectId(nextState, action.projectId);

      if (!projectId) {
        return;
      }

      recordDiagnosticEntry({
        context: action.context,
        level: 'error',
        message: action.message,
        namespace: action.namespace ?? 'system',
        source: { area: action.area ?? 'runtime', kind: 'workbench', projectId },
      });
      break;
    }
    case 'recordWidgetFailure': {
      if (Object.is(previousState, nextState)) {
        return;
      }

      const projectId = getDiagnosticProjectId(nextState);

      if (!projectId) {
        return;
      }

      recordDiagnosticEntry({
        context: { widgetId: action.failure.widgetId },
        level: 'error',
        message: action.failure.details,
        namespace: 'system',
        source: { area: 'widget-failure', kind: 'workbench', projectId },
      });
      break;
    }
    case 'closeProject': {
      if (previousState.projects.length !== 1) {
        return;
      }

      const projectId = getDiagnosticProjectId(nextState, action.projectId);

      if (!projectId) {
        return;
      }

      recordDiagnosticEntry({
        level: 'error',
        message: 'At least one project must remain open.',
        namespace: 'system',
        source: { area: 'project-lifecycle', kind: 'workbench', projectId },
      });
      break;
    }
  }
};

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
      const previousState = state;
      const nextState = workbenchReducer(state, action);

      setSnapshotState(nextState);
      recordDiagnosticForAction(action, previousState, nextState);
    },
    getPersistedRevision: () => persistedRevision,
    getSnapshot: snapshotStore.getSnapshot,
    getState: () => state,
    setHasHydrated: (nextHasHydrated) => setSnapshotState(state, nextHasHydrated),
    subscribe: snapshotStore.subscribe,
  };
};

export type { WorkbenchAction };
