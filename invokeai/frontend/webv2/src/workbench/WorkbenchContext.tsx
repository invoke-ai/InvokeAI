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

import { localStorageWorkbenchPersistence } from './persistence';
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

export const WorkbenchProvider = ({ children }: { children: ReactNode }) => {
  const [state, dispatch] = useReducer(workbenchReducer, undefined, createInitialWorkbenchState);
  const [hasHydrated, setHasHydrated] = useState(false);
  const hasLoadedPersistenceRef = useRef(false);
  const latestStateRef = useRef(state);
  const lastSavedStateKeyRef = useRef(getPersistedStateKey(state));
  const persistedStateKey = getPersistedStateKey(state);

  latestStateRef.current = state;

  useEffect(() => {
    let isCancelled = false;

    const loadPersistedState = async () => {
      try {
        const snapshot = await localStorageWorkbenchPersistence.loadWorkbench();

        if (isCancelled) {
          return;
        }

        if (snapshot) {
          lastSavedStateKeyRef.current = getPersistedStateKey(snapshot.state);
          dispatch({ state: snapshot.state, type: 'hydrateWorkbench' });
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
      localStorageWorkbenchPersistence
        .saveWorkbench(latestStateRef.current)
        .then((snapshot) => {
          if (isStale) {
            return;
          }

          lastSavedStateKeyRef.current = persistedStateKey;
          dispatch({ savedAt: snapshot.savedAt, type: 'autosaveSucceeded' });
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

  const value = useMemo<WorkbenchContextValue>(() => {
    const activeProject = state.projects.find((project) => project.id === state.activeProjectId) ?? state.projects[0];

    return { state, activeProject, dispatch, hasHydrated };
  }, [state, hasHydrated]);

  return <WorkbenchContext value={value}>{children}</WorkbenchContext>;
};

export const useWorkbench = (): WorkbenchContextValue => {
  const context = use(WorkbenchContext);

  if (!context) {
    throw new Error('useWorkbench must be used within a WorkbenchProvider.');
  }

  return context;
};
