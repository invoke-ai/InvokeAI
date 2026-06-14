import { createContext, use, useEffect, useRef, useSyncExternalStore, type Dispatch, type ReactNode } from 'react';

import {
  syncedWorkbenchPersistence,
  type WorkbenchLoadOptions,
  type WorkbenchSaveResult,
} from './projects/syncedPersistence';
import type { Project, WorkbenchState } from './types';
import { getAutosaveScheduleDecision } from './workbenchAutosave';
import {
  createWorkbenchStore,
  type WorkbenchAction,
  type WorkbenchSnapshot,
  type WorkbenchStore,
} from './workbenchStore';

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

type EqualityFn<T> = (left: T, right: T) => boolean;
type WorkbenchSelector<T> = (snapshot: WorkbenchSnapshot) => T;

const WorkbenchStoreContext = createContext<WorkbenchStore | null>(null);
const subscribeToNothing = (): (() => void) => () => {};
const getNullSnapshot = (): null => null;

export const shallowEqual = <T,>(left: T, right: T): boolean => {
  if (Object.is(left, right)) {
    return true;
  }

  if (typeof left !== 'object' || left === null || typeof right !== 'object' || right === null) {
    return false;
  }

  const leftRecord = left as Record<PropertyKey, unknown>;
  const rightRecord = right as Record<PropertyKey, unknown>;
  const leftKeys = Reflect.ownKeys(leftRecord);

  if (leftKeys.length !== Reflect.ownKeys(rightRecord).length) {
    return false;
  }

  return leftKeys.every(
    (key) => Object.prototype.hasOwnProperty.call(rightRecord, key) && Object.is(leftRecord[key], rightRecord[key])
  );
};

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
  const storeRef = useRef<WorkbenchStore | null>(null);

  if (storeRef.current === null) {
    storeRef.current = createWorkbenchStore();
  }

  const store = storeRef.current;
  const dispatch = store.dispatch;
  const hasLoadedPersistenceRef = useRef(false);
  const lastSavedStateKeyRef = useRef(getPersistedStateKey(store.getState()));
  // Captured once: the options describe how this mount of the editor boots.
  // Later search-param changes are handled live by WorkbenchSessionController.
  const bootOptionsRef = useRef(loadOptions);

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
          if (!syncedWorkbenchPersistence.hasPendingChanges()) {
            lastSavedStateKeyRef.current = getPersistedStateKey(snapshot.state);
          }

          dispatch({ state: snapshot.state, type: 'hydrateWorkbench' });
        }

        const requestedId = bootOptions?.openProjectId;
        const projects = snapshot?.state.projects ?? store.getState().projects;

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
          store.setHasHydrated(true);
        }
      }
    };

    void loadPersistedState();

    return () => {
      isCancelled = true;
    };
  }, [dispatch, store]);

  // Revision conflicts surfaced by a save are applied to state here: the
  // server version adopts the project id and the local edits continue in a
  // recovered fork. The follow-up autosave still persists the reconciled
  // session/cache, while project document pushes are no-ops because the sync
  // layer already acknowledged them.
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
    let timeoutId: number | null = null;
    let saveGeneration = 0;
    let failedStateKey: string | null = null;
    let scheduledStateKey: string | null = null;

    const clearScheduledSave = (): void => {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
        timeoutId = null;
      }
    };

    const scheduleSave = (): void => {
      if (!hasLoadedPersistenceRef.current) {
        return;
      }

      const state = store.getState();
      const persistedStateKey = getPersistedStateKey(state);
      const decision = getAutosaveScheduleDecision({
        failedStateKey,
        lastSavedStateKey: lastSavedStateKeyRef.current,
        persistedStateKey,
        scheduledStateKey,
      });

      failedStateKey = decision.failedStateKey;

      if (!decision.shouldSchedule) {
        return;
      }

      scheduledStateKey = persistedStateKey;
      saveGeneration += 1;
      const generation = saveGeneration;

      dispatch({ type: 'autosaveStarted' });
      clearScheduledSave();

      timeoutId = window.setTimeout(() => {
        const stateToSave = store.getState();
        const stateKeyToSave = getPersistedStateKey(stateToSave);

        syncedWorkbenchPersistence
          .saveWorkbench(stateToSave)
          .then((result) => {
            if (generation !== saveGeneration) {
              return;
            }

            lastSavedStateKeyRef.current = stateKeyToSave;
            failedStateKey = null;
            scheduledStateKey = null;
            dispatch({ savedAt: result.snapshot.savedAt, type: 'autosaveSucceeded' });
            applySaveResultRef.current(result);
          })
          .catch((error: unknown) => {
            if (generation !== saveGeneration) {
              return;
            }

            failedStateKey = stateKeyToSave;
            scheduledStateKey = null;
            dispatch({
              error: error instanceof Error ? error.message : 'Failed to autosave workbench.',
              type: 'autosaveFailed',
            });
          });
      }, AUTOSAVE_DELAY_MS);
    };

    const unsubscribe = store.subscribe(scheduleSave);
    scheduleSave();

    return () => {
      unsubscribe();
      saveGeneration += 1;
      clearScheduledSave();
    };
  }, [dispatch, store]);

  // Replay changes that queued up while the backend was unreachable as soon
  // as the socket reports it is back.
  useEffect(() => {
    let previousStatus = store.getState().backendConnection.status;

    return store.subscribe(() => {
      const status = store.getState().backendConnection.status;

      if (status === previousStatus) {
        return;
      }

      previousStatus = status;

      if (
        status !== 'connected' ||
        !hasLoadedPersistenceRef.current ||
        !syncedWorkbenchPersistence.hasPendingChanges()
      ) {
        return;
      }

      void syncedWorkbenchPersistence.saveWorkbench(store.getState()).then((result) => {
        applySaveResultRef.current(result);
      });
    });
  }, [store]);

  return (
    <WorkbenchStoreContext value={store}>
      <WorkbenchDispatchContext value={dispatch}>{children}</WorkbenchDispatchContext>
    </WorkbenchStoreContext>
  );
};

export const useWorkbenchStore = (): WorkbenchStore => {
  const store = use(WorkbenchStoreContext);

  if (!store) {
    throw new Error('useWorkbenchStore must be used within a WorkbenchProvider.');
  }

  return store;
};

export const useOptionalWorkbenchStore = (): WorkbenchStore | null => use(WorkbenchStoreContext);

export const useWorkbenchSelector = <Selected,>(
  selector: WorkbenchSelector<Selected>,
  isEqual: EqualityFn<Selected> = Object.is
): Selected => {
  const store = useWorkbenchStore();
  const selectionRef = useRef<Selected | undefined>(undefined);
  const hasSelectionRef = useRef(false);

  const getSelectedSnapshot = (): Selected => {
    const next = selector(store.getSnapshot());

    if (hasSelectionRef.current && isEqual(selectionRef.current as Selected, next)) {
      return selectionRef.current as Selected;
    }

    hasSelectionRef.current = true;
    selectionRef.current = next;

    return next;
  };

  return useSyncExternalStore(store.subscribe, getSelectedSnapshot, getSelectedSnapshot);
};

export const useActiveProject = (): Project => useWorkbenchSelector((snapshot) => snapshot.activeProject);

export const useActiveProjectSelector = <Selected,>(
  selector: (project: Project) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected => useWorkbenchSelector((snapshot) => selector(snapshot.activeProject), isEqual);

export const useWorkbenchHasHydrated = (): boolean => useWorkbenchSelector((snapshot) => snapshot.hasHydrated);

export const useOptionalWorkbenchSelector = <Selected,>(
  selector: WorkbenchSelector<Selected>,
  fallback: Selected,
  isEqual: EqualityFn<Selected> = Object.is
): Selected => {
  const store = useOptionalWorkbenchStore();
  const selectionRef = useRef<Selected | undefined>(undefined);
  const hasSelectionRef = useRef(false);

  const getSelectedSnapshot = (): Selected => {
    const next = store ? selector(store.getSnapshot()) : fallback;

    if (hasSelectionRef.current && isEqual(selectionRef.current as Selected, next)) {
      return selectionRef.current as Selected;
    }

    hasSelectionRef.current = true;
    selectionRef.current = next;

    return next;
  };

  return useSyncExternalStore(store?.subscribe ?? subscribeToNothing, getSelectedSnapshot, getSelectedSnapshot);
};

export const useOptionalWorkbenchDispatch = (): Dispatch<WorkbenchAction> | null =>
  useOptionalWorkbenchStore()?.dispatch ?? null;

export const useWorkbench = (): WorkbenchContextValue => {
  const store = useWorkbenchStore();
  const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);

  return { ...snapshot, dispatch: store.dispatch };
};

export const useOptionalWorkbench = (): WorkbenchContextValue | null => {
  const store = useOptionalWorkbenchStore();
  const snapshot = useSyncExternalStore(
    store?.subscribe ?? subscribeToNothing,
    store?.getSnapshot ?? getNullSnapshot,
    store?.getSnapshot ?? getNullSnapshot
  );

  return store && snapshot ? { ...snapshot, dispatch: store.dispatch } : null;
};

export const useWorkbenchDispatch = (): Dispatch<WorkbenchAction> => {
  const dispatch = use(WorkbenchDispatchContext);

  if (!dispatch) {
    throw new Error('useWorkbenchDispatch must be used within a WorkbenchProvider.');
  }

  return dispatch;
};
