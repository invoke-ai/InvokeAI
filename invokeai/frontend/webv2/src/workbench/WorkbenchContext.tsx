import {
  createContext,
  use,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
  type Dispatch,
  type ReactNode,
} from 'react';

import {
  syncedWorkbenchPersistence,
  type WorkbenchLoadOptions,
  type WorkbenchSaveResult,
} from './projects/syncedPersistence';
import type { Project, WorkbenchState } from './types';
import { createInitialWorkbenchState, workbenchReducer, type WorkbenchAction } from './workbenchState';

interface WorkbenchContextValue {
  state: WorkbenchState;
  activeProject: Project;
  dispatch: Dispatch<WorkbenchAction>;
  /**
   * True once the persisted snapshot has been loaded (or found absent). Side
   * effects that read or mutate the queue must wait for this, or they race the
   * async hydration and act on state that is about to be replaced.
   */
  hasHydrated: boolean;
}

const WorkbenchContext = createContext<WorkbenchContextValue | null>(null);

/**
 * The reducer dispatch alone, on its own context. `dispatch` is stable for the
 * provider's lifetime, so subscribers never re-render on state changes — the
 * subscription of choice for hot, many-instance components (e.g. flow nodes)
 * that only write.
 */
const WorkbenchDispatchContext = createContext<Dispatch<WorkbenchAction> | null>(null);

const AUTOSAVE_DELAY_MS = 500;

const getPersistedStateKey = (state: WorkbenchState): string =>
  JSON.stringify({
    account: state.account,
    activeProjectId: state.activeProjectId,
    errorLog: state.errorLog,
    notifications: state.notifications,
    projects: state.projects,
    widgetFailures: state.widgetFailures,
  });

export const WorkbenchProvider = ({
  children,
  loadOptions,
}: {
  children: ReactNode;
  /** Boot-time session options (deep-linked project, fresh draft). Read once at mount. */
  loadOptions?: WorkbenchLoadOptions;
}) => {
  const [state, dispatch] = useReducer(workbenchReducer, undefined, createInitialWorkbenchState);
  const [hasHydrated, setHasHydrated] = useState(false);
  const hasLoadedPersistenceRef = useRef(false);
  const latestStateRef = useRef(state);
  const lastSavedStateKeyRef = useRef(getPersistedStateKey(state));
  // Captured once: the options describe how this mount of the editor boots.
  // Later search-param changes are handled live by WorkbenchSessionController.
  const bootOptionsRef = useRef(loadOptions);
  const persistedStateKey = getPersistedStateKey(state);

  latestStateRef.current = state;

  useEffect(() => {
    let isCancelled = false;

    const loadPersistedState = async () => {
      const bootOptions = bootOptionsRef.current;

      try {
        const snapshot = await syncedWorkbenchPersistence.loadWorkbench(bootOptions);

        if (isCancelled) {
          return;
        }

        if (snapshot) {
          lastSavedStateKeyRef.current = getPersistedStateKey(snapshot.state);
          dispatch({ state: snapshot.state, type: 'hydrateWorkbench' });
        }

        const requestedId = bootOptions?.openProjectId;
        const projects = snapshot?.state.projects ?? latestStateRef.current.projects;

        if (requestedId && !projects.some((project) => project.id === requestedId)) {
          dispatch({
            kind: 'info',
            message: 'The linked project does not exist on this account — it may have been deleted.',
            title: 'Project not found',
            type: 'recordNotice',
          });
        }
      } catch (error) {
        dispatch({
          message: error instanceof Error ? error.message : 'Failed to load persisted workbench.',
          type: 'recordError',
        });
      } finally {
        hasLoadedPersistenceRef.current = true;

        if (!isCancelled) {
          setHasHydrated(true);
        }
      }
    };

    void loadPersistedState();

    return () => {
      isCancelled = true;
    };
  }, []);

  // Revision conflicts surfaced by a save are applied to state here: the
  // server version adopts the project id and the local edits continue in a
  // recovered fork. The follow-up autosave is a no-op for both (the sync
  // layer already acknowledged them).
  const applySaveResult = (result: WorkbenchSaveResult): void => {
    for (const conflict of result.conflicts) {
      dispatch({
        projectId: conflict.projectId,
        recoveredProject: conflict.recoveredProject,
        serverProject: conflict.serverProject,
        type: 'reconcileProjectConflict',
      });
    }
  };

  const applySaveResultRef = useRef(applySaveResult);

  applySaveResultRef.current = applySaveResult;

  useEffect(() => {
    if (!hasLoadedPersistenceRef.current) {
      return undefined;
    }

    if (persistedStateKey === lastSavedStateKeyRef.current) {
      return undefined;
    }

    let isStale = false;

    dispatch({ type: 'autosaveStarted' });

    const timeoutId = window.setTimeout(() => {
      syncedWorkbenchPersistence
        .saveWorkbench(latestStateRef.current)
        .then((result) => {
          if (isStale) {
            return;
          }

          lastSavedStateKeyRef.current = persistedStateKey;
          dispatch({ savedAt: result.snapshot.savedAt, type: 'autosaveSucceeded' });
          applySaveResultRef.current(result);
        })
        .catch((error: unknown) => {
          if (isStale) {
            return;
          }

          dispatch({
            error: error instanceof Error ? error.message : 'Failed to autosave workbench.',
            type: 'autosaveFailed',
          });
        });
    }, AUTOSAVE_DELAY_MS);

    return () => {
      isStale = true;
      window.clearTimeout(timeoutId);
    };
  }, [persistedStateKey]);

  // Replay changes that queued up while the backend was unreachable as soon
  // as the socket reports it is back.
  const backendConnectionStatus = state.backendConnection.status;

  useEffect(() => {
    if (
      backendConnectionStatus !== 'connected' ||
      !hasLoadedPersistenceRef.current ||
      !syncedWorkbenchPersistence.hasPendingChanges()
    ) {
      return;
    }

    void syncedWorkbenchPersistence.saveWorkbench(latestStateRef.current).then((result) => {
      applySaveResultRef.current(result);
    });
  }, [backendConnectionStatus]);

  const value = useMemo<WorkbenchContextValue>(() => {
    const activeProject = state.projects.find((project) => project.id === state.activeProjectId) ?? state.projects[0];

    return { state, activeProject, dispatch, hasHydrated };
  }, [state, hasHydrated]);

  return (
    <WorkbenchContext value={value}>
      <WorkbenchDispatchContext value={dispatch}>{children}</WorkbenchDispatchContext>
    </WorkbenchContext>
  );
};

export const useWorkbench = (): WorkbenchContextValue => {
  const context = use(WorkbenchContext);

  if (!context) {
    throw new Error('useWorkbench must be used within a WorkbenchProvider.');
  }

  return context;
};

export const useOptionalWorkbench = (): WorkbenchContextValue | null => use(WorkbenchContext);

export const useWorkbenchDispatch = (): Dispatch<WorkbenchAction> => {
  const dispatch = use(WorkbenchDispatchContext);

  if (!dispatch) {
    throw new Error('useWorkbenchDispatch must be used within a WorkbenchProvider.');
  }

  return dispatch;
};
