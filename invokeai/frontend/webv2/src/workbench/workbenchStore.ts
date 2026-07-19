import type { ProjectGraphAction } from '@features/workflow/utility';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { createExternalStore } from '@platform/state/externalStore';

import type { CanvasProjectMutation } from './canvasProjectMutations';

import { recordDiagnosticEntry } from './diagnostics/logger';
import {
  createInitialWorkbenchState,
  __workbenchReducerInternal,
  type __WorkbenchReducerActionInternal,
} from './workbenchState';

type WorkbenchAction = __WorkbenchReducerActionInternal;
type ActionPayload<Type extends WorkbenchAction['type']> = Omit<Extract<WorkbenchAction, { type: Type }>, 'type'>;
type WorkbenchDispatch = (action: WorkbenchAction) => void;

type MechanicalCommand<Type extends WorkbenchAction['type']> = keyof ActionPayload<Type> extends never
  ? () => void
  : (payload: ActionPayload<Type>) => void;

/**
 * Mechanical commands forward one action to the reducer; their public types derive
 * from the reducer's action union so the two cannot drift. The action union itself
 * stays private — callers only ever see the named command.
 */
const createCommandFactory = (dispatch: WorkbenchDispatch) => {
  function command<Type extends WorkbenchAction['type']>(type: Type): MechanicalCommand<Type>;
  function command<Type extends WorkbenchAction['type'], Args extends unknown[]>(
    type: Type,
    toPayload: (...args: Args) => ActionPayload<Type>
  ): (...args: Args) => void;
  function command(
    type: WorkbenchAction['type'],
    toPayload?: (...args: unknown[]) => Record<string, unknown>
  ): (...args: unknown[]) => void {
    return (...args) => {
      const payload = toPayload ? toPayload(...args) : ((args[0] ?? {}) as Record<string, unknown>);
      dispatch({ ...payload, type } as WorkbenchAction);
    };
  }

  return command;
};

export type ProjectCommandResult =
  | { ok: true }
  | { ok: false; reason: 'invalid-name' | 'last-project' | 'project-not-found' };

const createCommands = (dispatch: WorkbenchDispatch, getState: () => WorkbenchState) => {
  const command = createCommandFactory(dispatch);

  return {
    account: {
      updateProjectPreferences: command(
        'setActiveProjectSettings',
        (settings: ActionPayload<'setActiveProjectSettings'>['settings']) => ({ settings })
      ),
    },
    canvas: {
      apply: (projectId: string, mutation: CanvasProjectMutation): boolean => {
        const before = getState().projects.find((project) => project.id === projectId)?.canvas;
        if (!before) {
          return false;
        }

        dispatch({ mutation, projectId, type: 'applyCanvasProjectMutation' });
        return getState().projects.find((project) => project.id === projectId)?.canvas !== before;
      },
      appendStagingCandidate: command('appendCanvasStagingCandidate'),
    },
    gallery: {
      removeImages: command('removeGalleryImages', (imageNames: string[], projectId?: string) => ({
        imageNames,
        projectId,
      })),
      selectBoard: command('selectGalleryBoard', (boardId: string, projectId?: string) => ({ boardId, projectId })),
      selectImage: command(
        'selectGalleryImage',
        (image: ActionPayload<'selectGalleryImage'>['image'], projectId?: string) => ({ image, projectId })
      ),
      setCompareImage: command(
        'setGalleryCompareImage',
        (image: ActionPayload<'setGalleryCompareImage'>['image'], projectId?: string) => ({ image, projectId })
      ),
      setMultiSelection: command(
        'setGalleryMultiSelection',
        (
          imageNames: string[],
          primaryImage: ActionPayload<'setGalleryMultiSelection'>['primaryImage'],
          projectId?: string
        ) => ({ imageNames, primaryImage, projectId })
      ),
      setPage: command('setGalleryPage', (page: number, projectId?: string) => ({ page, projectId })),
      setPageInfo: command('setGalleryPageInfo', (totalImages: number, projectId?: string) => ({
        projectId,
        totalImages,
      })),
      setProjectBoard: command('setGalleryProjectBoardId', (boardId: string, projectId?: string) => ({
        boardId,
        projectId,
      })),
      setSearchTerm: command('setGallerySearchTerm', (searchTerm: string, projectId?: string) => ({
        projectId,
        searchTerm,
      })),
      setView: command(
        'setGalleryView',
        (galleryView: ActionPayload<'setGalleryView'>['galleryView'], projectId?: string) => ({
          galleryView,
          projectId,
        })
      ),
      toggleImageSelection: command(
        'toggleGalleryImageInSelection',
        (image: ActionPayload<'toggleGalleryImageInSelection'>['image'], projectId?: string) => ({
          image,
          projectId,
        })
      ),
      touch: command('touchGalleryRefresh', (projectId?: string) => ({ projectId })),
      touchImages: command('touchGalleryImagesRefresh', (projectId?: string) => ({ projectId })),
      updateSettings: command(
        'updateGallerySettings',
        (settings: ActionPayload<'updateGallerySettings'>['settings'], projectId?: string) => ({
          projectId,
          settings,
        })
      ),
    },
    generation: {
      clearPromptHistory: command('clearPromptHistory', (projectId?: string) => ({ projectId })),
      patchPromptDraft: command(
        'patchProjectPromptDraft',
        (values: ActionPayload<'patchProjectPromptDraft'>['values'], projectId?: string) => ({ projectId, values })
      ),
      patchSettings: command(
        'patchGenerateSettings',
        (values: ActionPayload<'patchGenerateSettings'>['values'], projectId?: string) => ({ projectId, values })
      ),
      removePromptFromHistory: command(
        'removePromptFromHistory',
        (prompt: ActionPayload<'removePromptFromHistory'>['prompt'], projectId?: string) => ({ projectId, prompt })
      ),
      setBatchCount: command('setGenerateBatchCount', (batchCount: number, projectId?: string) => ({
        batchCount,
        projectId,
      })),
      setDestination: command(
        'setInvocationDestination',
        (destination: ActionPayload<'setInvocationDestination'>['destination']) => ({ destination })
      ),
      setSettings: command(
        'setGenerateSettings',
        (values: ActionPayload<'setGenerateSettings'>['values'], projectId?: string) => ({ projectId, values })
      ),
      setSource: command('setInvocationSource', (sourceId: ActionPayload<'setInvocationSource'>['sourceId']) => ({
        sourceId,
      })),
      submitCanvas: command('submitCanvasInvocationSnapshot'),
      submitResolved: command('submitResolvedInvocationSnapshot'),
      toggleDestinationLock: command('toggleDestinationLock'),
      toggleSourceLock: command('toggleSourceLock'),
    },
    layout: {
      applyPreset: command('applyPreset', (presetId: ActionPayload<'applyPreset'>['presetId']) => ({ presetId })),
      createPreset: command(
        'addLayoutPreset',
        (presetId: ActionPayload<'addLayoutPreset'>['presetId'], label: string) => ({
          label,
          presetId,
        })
      ),
      deletePreset: command('deleteLayoutPreset', (presetId: ActionPayload<'deleteLayoutPreset'>['presetId']) => ({
        presetId,
      })),
      recover: command('recoverShellLayout'),
      renamePreset: command(
        'renameLayoutPreset',
        (presetId: ActionPayload<'renameLayoutPreset'>['presetId'], label: string) => ({ label, presetId })
      ),
      reset: command('resetActiveLayout'),
      setCenterView: command('setCenterView', (centerViewId: ActionPayload<'setCenterView'>['centerViewId']) => ({
        centerViewId,
      })),
      setRegionCollapsed: command(
        'setRegionWidgetCollapsed',
        (region: ActionPayload<'setRegionWidgetCollapsed'>['region'], isCollapsed: boolean) => ({
          isCollapsed,
          region,
        })
      ),
      setRegionSize: command(
        'setRegionWidgetSize',
        (region: ActionPayload<'setRegionWidgetSize'>['region'], sizePx: number) => ({ region, sizePx })
      ),
    },
    notifications: {
      add: command('recordNotice'),
      clear: command('clearNotifications'),
      markAllRead: command('markAllNotificationsRead'),
      recordWidgetFailure: command(
        'recordWidgetFailure',
        (failure: ActionPayload<'recordWidgetFailure'>['failure']) => ({ failure })
      ),
      reportError: command('recordError'),
    },
    projects: {
      close: (projectId: string): ProjectCommandResult => {
        if (!getState().projects.some((project) => project.id === projectId)) {
          return { ok: false, reason: 'project-not-found' };
        }

        if (getState().projects.length === 1) {
          dispatch({ projectId, type: 'closeProject' });
          return { ok: false, reason: 'last-project' };
        }

        dispatch({ projectId, type: 'closeProject' });
        return { ok: true };
      },
      create: (): Project => {
        dispatch({ type: 'createProject' });
        return getActiveProject(getState());
      },
      open: command('openProject', (project: Project) => ({ project })),
      rename: (projectId: string, name: string): ProjectCommandResult => {
        if (!getState().projects.some((project) => project.id === projectId)) {
          return { ok: false, reason: 'project-not-found' };
        }
        if (!name.trim()) {
          return { ok: false, reason: 'invalid-name' };
        }

        dispatch({ name, projectId, type: 'renameProject' });
        return { ok: true };
      },
      switchTo: (projectId: string): ProjectCommandResult => {
        if (!getState().projects.some((project) => project.id === projectId)) {
          return { ok: false, reason: 'project-not-found' };
        }

        dispatch({ projectId, type: 'switchProject' });
        return { ok: true };
      },
    },
    queue: {
      cancel: command('cancelQueueItem', (projectId: string | undefined, queueItemId: string) => ({
        projectId,
        queueItemId,
      })),
      cancelAll: command('cancelAllQueueItems', (projectId?: string) => ({ projectId })),
      cancelAllExceptCurrent: command(
        'cancelAllQueueItemsExceptCurrent',
        (projectId?: string, currentQueueItemId?: string | null) => ({ currentQueueItemId, projectId })
      ),
      clearCompleted: command('clearCompletedQueueItems'),
      markBackendCancelled: command('markQueueItemBackendCancelled'),
      markBackendSubmitted: command('markQueueItemBackendSubmitted'),
      refreshBackendData: command('refreshBackendData'),
      routePartialResults: command('routeQueueItemPartialResults'),
      routeResults: command('routeQueueItemResults'),
      setConnectionStatus: command('setBackendConnectionStatus'),
      setStatus: command('setQueueItemStatus'),
    },
    widgets: {
      move: command('moveWidgetInstance'),
      open: command('openRegionWidget'),
      patchInstanceValues: command(
        'patchWidgetInstanceValues',
        (instanceId: string, values: Record<string, unknown>, projectId?: string) => ({
          instanceId,
          projectId,
          values,
        })
      ),
      patchValues: command(
        'patchWidgetValues',
        (
          widgetId: ActionPayload<'patchWidgetValues'>['widgetId'],
          values: Record<string, unknown>,
          projectId?: string
        ) => ({
          projectId,
          values,
          widgetId,
        })
      ),
      reorder: command('reorderWidgetInstances'),
      select: command('selectRegionWidget'),
      setInstanceValues: command(
        'setWidgetInstanceValues',
        (instanceId: string, values: Record<string, unknown>, projectId?: string) => ({
          instanceId,
          projectId,
          values,
        })
      ),
      toggle: command('toggleRegionWidget'),
    },
    workflows: {
      bindLibraryWorkflow: command('setProjectGraphLibraryBinding', (libraryWorkflowId: string) => ({
        libraryWorkflowId,
      })),
      editGraph: command('applyProjectGraphAction', (action: ProjectGraphAction) => ({ action })),
      replace: command(
        'replaceProjectGraph',
        (document: ActionPayload<'replaceProjectGraph'>['document'], label: string) => ({ document, label })
      ),
      redo: command('redoProjectChange'),
      restoreSnapshot: command('restoreProjectGraphSnapshot', (snapshotId: string) => ({ snapshotId })),
      saveSnapshot: command('saveProjectGraphSnapshot'),
      undo: command('undoProjectChange'),
    },
  };
};

const createPersistenceAdapter = (dispatch: WorkbenchDispatch, getState: () => WorkbenchState) => {
  const command = createCommandFactory(dispatch);

  return {
    getState,
    hydrate: command('hydrateWorkbench', (state: WorkbenchState) => ({ state })),
    reconcileConflict: command('reconcileProjectConflict'),
    saveFailed: command('autosaveFailed', (error: string) => ({ error })),
    saveStarted: command('autosaveStarted'),
    saveSucceeded: command('autosaveSucceeded', (savedAt: string) => ({ savedAt })),
  };
};

export type WorkbenchCommands = ReturnType<typeof createCommands>;
export type WorkbenchAccountCommands = WorkbenchCommands['account'];
export type WorkbenchCanvasCommands = WorkbenchCommands['canvas'];
export type WorkbenchGalleryCommands = WorkbenchCommands['gallery'];
export type WorkbenchGenerationCommands = WorkbenchCommands['generation'];
export type WorkbenchLayoutCommands = WorkbenchCommands['layout'];
export type WorkbenchNotificationCommands = WorkbenchCommands['notifications'];
export type WorkbenchProjectCommands = WorkbenchCommands['projects'];
export type WorkbenchQueueCommands = WorkbenchCommands['queue'];
export type WorkbenchWidgetCommands = WorkbenchCommands['widgets'];
export type WorkbenchWorkflowCommands = WorkbenchCommands['workflows'];

export type WorkbenchPersistenceCommands = ReturnType<typeof createPersistenceAdapter>;

export interface WorkbenchInternalAdapters {
  persistence: WorkbenchPersistenceCommands;
}

export interface WorkbenchSnapshot {
  activeProject: Project;
  account: WorkbenchState['account'];
  autosave: WorkbenchState['autosave'];
  backendConnection: WorkbenchState['backendConnection'];
  hasHydrated: boolean;
  notifications: WorkbenchState['notifications'];
  projects: WorkbenchState['projects'];
}

/** Imperative, read-only views for event handlers that must observe post-flush state. */
export interface WorkbenchQueries {
  getProject(projectId: string): Project | null;
  getSnapshot(): WorkbenchSnapshot;
  isActiveProject(projectId: string): boolean;
}

export interface WorkbenchInternalStore {
  commands: WorkbenchCommands;
  getPersistedRevision: () => number;
  getSnapshot: () => WorkbenchSnapshot;
  getState: () => WorkbenchState;
  /** Privileged implementation adapters; never exposed through React hooks. */
  internal: WorkbenchInternalAdapters;
  queries: WorkbenchQueries;
  setHasHydrated: (hasHydrated: boolean) => void;
  subscribe: (listener: () => void) => () => void;
}

const getActiveProject = (state: WorkbenchState): Project =>
  state.projects.find((project) => project.id === state.activeProjectId) ?? state.projects[0];

const createSnapshot = (state: WorkbenchState, hasHydrated: boolean): WorkbenchSnapshot => ({
  account: state.account,
  activeProject: getActiveProject(state),
  autosave: state.autosave,
  backendConnection: state.backendConnection,
  hasHydrated,
  notifications: state.notifications,
  projects: state.projects,
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

export const createWorkbenchStore = (initialState = createInitialWorkbenchState()): WorkbenchInternalStore => {
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

  const dispatch = (action: WorkbenchAction): void => {
    const previousState = state;
    const nextState = __workbenchReducerInternal(state, action);

    setSnapshotState(nextState);
    recordDiagnosticForAction(action, previousState, nextState);
  };

  const getState = (): WorkbenchState => state;

  return {
    commands: createCommands(dispatch, getState),
    getPersistedRevision: () => persistedRevision,
    getSnapshot: snapshotStore.getSnapshot,
    getState,
    internal: {
      persistence: createPersistenceAdapter(dispatch, getState),
    },
    queries: {
      getProject: (projectId) => state.projects.find((project) => project.id === projectId) ?? null,
      getSnapshot: snapshotStore.getSnapshot,
      isActiveProject: (projectId) => state.activeProjectId === projectId,
    },
    setHasHydrated: (nextHasHydrated) => setSnapshotState(state, nextHasHydrated),
    subscribe: snapshotStore.subscribe,
  };
};
