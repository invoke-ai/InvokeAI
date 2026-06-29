import {
  createContext,
  use,
  useEffect,
  useEffectEvent,
  useRef,
  useSyncExternalStore,
  useState,
  type Dispatch,
  type ReactNode,
} from 'react';

import type {
  Project,
  ProjectLayoutState,
  ProjectSettings,
  WidgetInstanceId,
  WidgetTypeId,
  WorkbenchState,
} from './types';

import { WorkbenchSplashScreen } from './components/WorkbenchSplashScreen';
import {
  syncedWorkbenchPersistence,
  type WorkbenchLoadOptions,
  type WorkbenchSaveResult,
} from './projects/syncedPersistence';
import { getProjectWidgetValues } from './widgetState';
import { getAutosaveScheduleDecision, isAutosaveCompletionCurrent } from './workbenchAutosave';
import { shallowEqual as selectorShallowEqual, useExternalStoreSelector } from './workbenchSelectors';
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

export const shallowEqual = selectorShallowEqual;

/**
 * The reducer dispatch alone, on its own context. `dispatch` is stable for the
 * provider's lifetime, so subscribers never re-render on state changes — the
 * subscription of choice for hot, many-instance components (e.g. flow nodes)
 * that only write.
 */
const WorkbenchDispatchContext = createContext<Dispatch<WorkbenchAction> | null>(null);

const AUTOSAVE_DELAY_MS = 500;

export const WorkbenchProvider = ({
  children,
  loadOptions,
}: {
  children: ReactNode;
  /** Boot-time session options (deep-linked project, fresh draft). Read once at mount. */
  loadOptions?: WorkbenchLoadOptions;
}) => {
  const [store] = useState(() => createWorkbenchStore());
  const dispatch = store.dispatch;
  const [hasHydrated, setHasHydrated] = useState(store.getSnapshot().hasHydrated);
  const hasLoadedPersistenceRef = useRef(false);
  const lastSavedPersistedRevisionRef = useRef(store.getPersistedRevision());
  const failedPersistedRevisionRef = useRef<number | null>(null);
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
          const isPendingSnapshot = syncedWorkbenchPersistence.hasPendingChanges();

          dispatch({ state: snapshot.state, type: 'hydrateWorkbench' });

          if (!isPendingSnapshot) {
            lastSavedPersistedRevisionRef.current = store.getPersistedRevision();
          }
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
          area: 'persistence-load',
          message: error instanceof Error ? error.message : 'Failed to load persisted workbench.',
          namespace: 'system',
          type: 'recordError',
        });
      } finally {
        hasLoadedPersistenceRef.current = true;

        if (!isCancelled) {
          store.setHasHydrated(true);
          setHasHydrated(true);
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
  const applySaveResult = useEffectEvent((result: WorkbenchSaveResult): void => {
    for (const conflict of result.conflicts) {
      dispatch({
        projectId: conflict.projectId,
        recoveredProject: conflict.recoveredProject,
        serverProject: conflict.serverProject,
        type: 'reconcileProjectConflict',
      });
    }
  });

  useEffect(() => {
    let timeoutId: number | null = null;
    let saveGeneration = 0;
    let scheduledPersistedRevision: number | null = null;

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

      const persistedRevision = store.getPersistedRevision();
      const decision = getAutosaveScheduleDecision({
        failedPersistedRevision: failedPersistedRevisionRef.current,
        lastSavedPersistedRevision: lastSavedPersistedRevisionRef.current,
        persistedRevision,
        scheduledPersistedRevision,
      });

      failedPersistedRevisionRef.current = decision.failedPersistedRevision;

      if (!decision.shouldSchedule) {
        return;
      }

      scheduledPersistedRevision = persistedRevision;
      saveGeneration += 1;
      const generation = saveGeneration;

      clearScheduledSave();

      timeoutId = window.setTimeout(() => {
        const stateToSave = store.getState();
        const revisionToSave = store.getPersistedRevision();

        dispatch({ type: 'autosaveStarted' });

        syncedWorkbenchPersistence
          .saveWorkbench(stateToSave)
          .then((result) => {
            if (generation !== saveGeneration) {
              return;
            }

            lastSavedPersistedRevisionRef.current = revisionToSave;
            failedPersistedRevisionRef.current = null;
            scheduledPersistedRevision = null;
            dispatch({ savedAt: result.snapshot.savedAt, type: 'autosaveSucceeded' });
            applySaveResult(result);
          })
          .catch((error: unknown) => {
            if (generation !== saveGeneration) {
              return;
            }

            failedPersistedRevisionRef.current = revisionToSave;
            scheduledPersistedRevision = null;
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

      const stateToSave = store.getState();
      const revisionToSave = store.getPersistedRevision();

      void syncedWorkbenchPersistence
        .saveWorkbench(stateToSave)
        .then((result) => {
          if (
            !isAutosaveCompletionCurrent({
              completedPersistedRevision: revisionToSave,
              currentPersistedRevision: store.getPersistedRevision(),
            })
          ) {
            return;
          }

          lastSavedPersistedRevisionRef.current = revisionToSave;
          failedPersistedRevisionRef.current = null;
          dispatch({ savedAt: result.snapshot.savedAt, type: 'autosaveSucceeded' });
          applySaveResult(result);
        })
        .catch((error: unknown) => {
          if (
            !isAutosaveCompletionCurrent({
              completedPersistedRevision: revisionToSave,
              currentPersistedRevision: store.getPersistedRevision(),
            })
          ) {
            return;
          }

          failedPersistedRevisionRef.current = revisionToSave;
          dispatch({
            error: error instanceof Error ? error.message : 'Failed to autosave workbench.',
            type: 'autosaveFailed',
          });
        });
    });
  }, [dispatch, store]);

  return (
    <WorkbenchStoreContext value={store}>
      <WorkbenchDispatchContext value={dispatch}>
        {hasHydrated ? children : <WorkbenchSplashScreen message="Opening project" />}
      </WorkbenchDispatchContext>
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
  isEqual: EqualityFn<Selected> = shallowEqual
): Selected => {
  const store = useWorkbenchStore();

  return useExternalStoreSelector(store.subscribe, store.getSnapshot, selector, isEqual);
};

export const useDebouncedWorkbenchSelector = <Selected,>(
  selector: WorkbenchSelector<Selected>,
  debounceMs = 300,
  isEqual: EqualityFn<Selected> = Object.is
): Selected => {
  const liveSelection = useWorkbenchSelector(selector, isEqual);
  const [selection, setSelection] = useState(liveSelection);

  useEffect(() => {
    if (isEqual(selection, liveSelection)) {
      return;
    }

    const timeoutId = window.setTimeout(() => {
      setSelection(liveSelection);
    }, debounceMs);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [debounceMs, isEqual, liveSelection, selection]);

  return selection;
};

export const useActiveProject = (): Project => useWorkbenchSelector((snapshot) => snapshot.activeProject);

export const useActiveProjectSelector = <Selected,>(
  selector: (project: Project) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected => useWorkbenchSelector((snapshot) => selector(snapshot.activeProject), isEqual);

export const useActiveProjectId = (): string => useWorkbenchSelector((snapshot) => snapshot.state.activeProjectId);

export const useActiveProjectName = (): string => useActiveProjectSelector((project) => project.name);

export const useActiveProjectLayoutSelector = <Selected,>(
  selector: (layout: ProjectLayoutState) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected => useActiveProjectSelector((project) => selector(project.layout), isEqual);

export const useActiveProjectSettingsSelector = <Selected,>(
  selector: (settings: ProjectSettings) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected => useActiveProjectSelector((project) => selector(project.settings), isEqual);

export const useWidgetValuesSelector = <Selected,>(
  widgetId: WidgetTypeId,
  selector: (values: Record<string, unknown>) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected => useActiveProjectSelector((project) => selector(getProjectWidgetValues(project, widgetId)), isEqual);

export const useWidgetInstanceValuesSelector = <Selected,>(
  instanceId: WidgetInstanceId,
  selector: (values: Record<string, unknown>) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected =>
  useActiveProjectSelector((project) => selector(project.widgetInstances[instanceId]?.state.values ?? {}), isEqual);

export const useProjectWidgetInstanceValuesSelector = <Selected,>(
  projectId: string,
  instanceId: WidgetInstanceId,
  selector: (values: Record<string, unknown>) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected =>
  useWorkbenchSelector((snapshot) => {
    const project = snapshot.state.projects.find((candidate) => candidate.id === projectId);

    return selector(project?.widgetInstances[instanceId]?.state.values ?? {});
  }, isEqual);

export const useWorkbenchHasHydrated = (): boolean => useWorkbenchSelector((snapshot) => snapshot.hasHydrated);

export const useOptionalWorkbenchSelector = <Selected,>(
  selector: WorkbenchSelector<Selected>,
  fallback: Selected,
  isEqual: EqualityFn<Selected> = shallowEqual
): Selected => {
  const store = useOptionalWorkbenchStore();

  return useExternalStoreSelector(
    store?.subscribe ?? subscribeToNothing,
    store?.getSnapshot ?? getNullSnapshot,
    (snapshot) => (snapshot ? selector(snapshot) : fallback),
    isEqual
  );
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
