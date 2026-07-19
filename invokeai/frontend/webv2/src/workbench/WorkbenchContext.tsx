import type { ProjectLayoutState } from '@workbench/layoutContracts';
import type { Project } from '@workbench/projectContracts';
import type { ProjectSettings } from '@workbench/settings/contracts';
import type { WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';

import { useMountEffect } from '@platform/react/useMountEffect';
import { shallowEqual as selectorShallowEqual, useExternalStoreSelector } from '@platform/state/selectors';
import { createContext, use, useEffect, useSyncExternalStore, useState, type ReactNode } from 'react';

import { WorkbenchSplashScreen } from './components/WorkbenchSplashScreen';
import { createWorkbenchPersistenceRuntime } from './persistenceRuntime';
import {
  createSyncedWorkbenchPersistence,
  type SyncedWorkbenchPersistence,
  type WorkbenchLoadOptions,
} from './projects/syncedPersistence';
import { getProjectWidgetValues } from './widgetState';
import { createWorkbenchStore, type WorkbenchSnapshot, type WorkbenchInternalStore } from './workbenchStore';

interface WorkbenchContextValue {
  activeProject: Project;
  /**
   * True once the persisted snapshot has been loaded (or found absent). Side
   * effects that read or mutate the queue must wait for this, or they race the
   * async hydration and act on state that is about to be replaced.
   */
  hasHydrated: boolean;
}

type EqualityFn<T> = (left: T, right: T) => boolean;
type WorkbenchSelector<T> = (snapshot: WorkbenchSnapshot) => T;

const WorkbenchStoreContext = createContext<WorkbenchInternalStore | null>(null);
const WorkbenchPersistenceContext = createContext<SyncedWorkbenchPersistence | null>(null);
const subscribeToNothing = (): (() => void) => () => {};
const getNullSnapshot = (): null => null;

export const shallowEqual = selectorShallowEqual;

export const WorkbenchProvider = ({
  children,
  loadOptions,
}: {
  children: ReactNode;
  /** Boot-time session options (deep-linked project, fresh draft). Read once at mount. */
  loadOptions?: WorkbenchLoadOptions;
}) => {
  const [store] = useState(() => createWorkbenchStore());
  const [persistence] = useState(createSyncedWorkbenchPersistence);
  const hasHydrated = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot).hasHydrated;

  // The runtime is created inside the effect: disposal is terminal, so each
  // mount (including a StrictMode remount) must get its own instance.
  useMountEffect(() => {
    const persistenceRuntime = createWorkbenchPersistenceRuntime({
      aggregate: {
        ...store.internal.persistence,
        getPersistedRevision: store.getPersistedRevision,
        notifyProjectNotFound: () =>
          store.commands.notifications.add({
            kind: 'info',
            message: 'The linked project does not exist on this account — it may have been deleted.',
            title: 'Project not found',
          }),
        reportLoadError: (message) =>
          store.commands.notifications.reportError({ area: 'persistence-load', message, namespace: 'system' }),
        setHasHydrated: store.setHasHydrated,
        subscribe: store.subscribe,
      },
      loadOptions,
      persistence,
    });
    persistenceRuntime.start();
    return () => persistenceRuntime.dispose();
  });

  return (
    <WorkbenchPersistenceContext value={persistence}>
      <WorkbenchStoreContext value={store}>
        {hasHydrated ? children : <WorkbenchSplashScreen messageKey="splash.openingProject" />}
      </WorkbenchStoreContext>
    </WorkbenchPersistenceContext>
  );
};

const useWorkbenchStore = (): WorkbenchInternalStore => {
  const store = use(WorkbenchStoreContext);

  if (!store) {
    throw new Error('useWorkbenchStore must be used within a WorkbenchProvider.');
  }

  return store;
};

const useOptionalWorkbenchStore = (): WorkbenchInternalStore | null => use(WorkbenchStoreContext);

/** Privileged aggregate adapter for persistence and resource-owning runtimes only. */
export const useWorkbenchInternalStore = (): WorkbenchInternalStore => useWorkbenchStore();

export const useHasWorkbenchProvider = (): boolean => useOptionalWorkbenchStore() !== null;

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

export const useActiveProjectId = (): string => useWorkbenchSelector((snapshot) => snapshot.activeProject.id);

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
    const project = snapshot.projects.find((candidate) => candidate.id === projectId);

    return selector(project?.widgetInstances[instanceId]?.state.values ?? {});
  }, isEqual);

export const useWorkbenchHasHydrated = (): boolean => useWorkbenchSelector((snapshot) => snapshot.hasHydrated);

/** Stable intent-oriented aggregate commands; callers never receive reducer actions. */
export const useWorkbenchCommands = () => useWorkbenchStore().commands;

export const useWorkbenchQueries = () => useWorkbenchStore().queries;

/** Stable read-model subscription used by external runtime adapters. */
export const useWorkbenchSubscription = () => useWorkbenchStore().subscribe;

/** Privileged persistence read/write port; this is the only UI-adjacent full-state adapter. */
export const useWorkbenchPersistenceAdapter = () => useWorkbenchStore().internal.persistence;

export const useWorkbenchPersistenceService = (): SyncedWorkbenchPersistence => {
  const persistence = use(WorkbenchPersistenceContext);
  if (!persistence) {
    throw new Error('useWorkbenchPersistenceService must be used within a WorkbenchProvider.');
  }
  return persistence;
};

export const useOptionalWorkbenchPersistenceService = (): SyncedWorkbenchPersistence | null =>
  use(WorkbenchPersistenceContext);

export const useOptionalWorkbenchCommands = () => useOptionalWorkbenchStore()?.commands ?? null;

export const useOptionalWorkbenchQueries = () => useOptionalWorkbenchStore()?.queries ?? null;

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

export const useWorkbench = (): WorkbenchContextValue => {
  const store = useWorkbenchStore();
  const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);

  return snapshot;
};

export const useOptionalWorkbench = (): WorkbenchContextValue | null => {
  const store = useOptionalWorkbenchStore();
  const snapshot = useSyncExternalStore(
    store?.subscribe ?? subscribeToNothing,
    store?.getSnapshot ?? getNullSnapshot,
    store?.getSnapshot ?? getNullSnapshot
  );

  return store && snapshot ? snapshot : null;
};
