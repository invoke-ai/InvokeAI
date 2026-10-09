import type { ProjectGraphState, ProjectWorkflowSource } from '@features/workflow/contracts';
import type { ProjectGraphAction } from '@features/workflow/utility';
import type { LayoutPreset } from '@workbench/layoutContracts';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import {
  legacyGeneratedImageToGalleryItem,
  type GalleryImage,
  type GeneratedImageContract,
} from '@features/gallery/contracts';
import { recordLogEvent } from '@platform/logging/logger';
import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { createExternalStore } from '@platform/state/externalStore';
import { closeWidgetOverlays } from '@platform/ui/widgetOverlayRegistry';
import { hasActiveQueueRuns, hasInFlightQueueRuns } from '@workbench/queue-integration/activeQueueRuns';

import type { CanvasEditIntent } from './autoRoutePolicy';
import type { CanvasProjectMutation } from './canvasProjectMutations';

import { clearLayerPanelStates, reconcileLayerPanelStates } from './layerPanelState';
import { createLayoutPresetActivator, loadLayoutPresetWidgets } from './layoutPresetActivation';
import { getLayoutPresetArrangement, resolveSavedLayoutPreset } from './layoutPresetSnapshots';
import { getLayoutWidgetTypeIds } from './layoutWidgetSet';
import { createBlankWorkflowDocument, findProjectWorkflow } from './projectWorkflows';
import { getWorkbenchPreferences } from './settings/store';
import { areWidgetsLoaded } from './widgetRegistry';
import { getProjectWidgetValues } from './widgetState';
import {
  createInitialWorkbenchState,
  __workbenchReducerInternal,
  type __WorkbenchReducerActionInternal,
} from './workbenchState';

/** Visible-widget changes dismiss overlays that would otherwise outlive their widgets in portals. */
const visibleWidgetsKey = (state: WorkbenchState): string => {
  const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);
  if (!project) {
    return '';
  }
  const regions = Object.entries(project.widgetRegions).map(
    ([region, { activeInstanceId }]) => `${region}=${activeInstanceId}`
  );
  return [project.id, ...regions, ...Object.keys(project.floatingWidgets ?? {})].join('|');
};

type WorkbenchAction = __WorkbenchReducerActionInternal;
type ActionPayload<Type extends WorkbenchAction['type']> = Omit<Extract<WorkbenchAction, { type: Type }>, 'type'>;
type WorkbenchDispatch = (action: WorkbenchAction) => void;

type MechanicalCommand<Type extends WorkbenchAction['type']> = keyof ActionPayload<Type> extends never
  ? () => void
  : (payload: ActionPayload<Type>) => void;

/** Derive named command types from the private reducer action union to prevent drift. */
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
  | {
      ok: false;
      reason:
        | 'active-queue-runs'
        | 'invalid-name'
        | 'last-project'
        | 'modified'
        | 'project-not-found'
        | 'target-already-open';
    };

const createCommands = (
  dispatch: WorkbenchDispatch,
  getState: () => WorkbenchState,
  layoutPresetActivator: Pick<ReturnType<typeof createLayoutPresetActivator>, 'activate' | 'invalidate'>
) => {
  const command = createCommandFactory(dispatch);

  return {
    account: {
      updateProjectPreferences: command(
        'setActiveProjectSettings',
        (settings: ActionPayload<'setActiveProjectSettings'>['settings']) => ({ settings })
      ),
    },
    canvas: {
      apply: (
        projectId: string,
        mutation: CanvasProjectMutation,
        origin?: ActionPayload<'applyCanvasProjectMutation'>['origin']
      ): boolean => {
        const before = getState().projects.find((project) => project.id === projectId)?.canvas;
        if (!before) {
          return false;
        }

        dispatch({ mutation, origin, projectId, type: 'applyCanvasProjectMutation' });
        return getState().projects.find((project) => project.id === projectId)?.canvas !== before;
      },
      commitEdit: (projectId: string, intent: CanvasEditIntent): void =>
        dispatch({ intent, projectId, type: 'commitCanvasEdit' }),
      appendStagingCandidate: command('appendCanvasStagingCandidate'),
    },
    gallery: {
      clearSelection: command('clearGallerySelection', (projectId?: string) => ({ projectId })),
      patchItems: command(
        'patchGalleryItems',
        (
          itemKeys: ActionPayload<'patchGalleryItems'>['itemKeys'],
          changes: ActionPayload<'patchGalleryItems'>['changes']
        ) => ({
          changes,
          itemKeys,
        })
      ),
      removeItems: command('removeGalleryItems', (itemKeys: ActionPayload<'removeGalleryItems'>['itemKeys']) => ({
        itemKeys,
      })),
      reconcileDeletedBoardOutcome: command(
        'reconcileDeletedGalleryBoard',
        (outcome: ActionPayload<'reconcileDeletedGalleryBoard'>['outcome']) => ({ outcome })
      ),
      selectItem: command(
        'selectGalleryItem',
        (
          item: ActionPayload<'selectGalleryItem'>['item'],
          projectId?: string,
          selectionPage?: number,
          preserveNavigationQuery?: boolean
        ) => ({
          item,
          preserveNavigationQuery,
          projectId,
          selectionPage,
        })
      ),
      setCompareItem: command(
        'setGalleryCompareImage',
        (image: ActionPayload<'setGalleryCompareImage'>['image'], projectId?: string) => ({ image, projectId })
      ),
      setItemMultiSelection: command(
        'setGalleryMultiSelection',
        (
          itemKeys: ActionPayload<'setGalleryMultiSelection'>['itemKeys'],
          primaryItem: ActionPayload<'setGalleryMultiSelection'>['primaryItem'],
          projectId?: string,
          selectionPage?: number,
          preserveNavigationQuery?: boolean
        ) => ({ itemKeys, preserveNavigationQuery, primaryItem, projectId, selectionPage })
      ),
      toggleItemSelection: command(
        'toggleGalleryItemInSelection',
        (
          item: ActionPayload<'toggleGalleryItemInSelection'>['item'],
          nextPrimaryItem: ActionPayload<'toggleGalleryItemInSelection'>['nextPrimaryItem'],
          projectId?: string,
          selectionPage?: number
        ) => ({
          item,
          nextPrimaryItem,
          projectId,
          selectionPage,
        })
      ),
      selectImage: (
        image: GeneratedImageContract & Partial<GalleryImage>,
        projectId?: string,
        selectionPage?: number,
        preserveNavigationQuery?: boolean
      ): void =>
        dispatch({
          item: legacyGeneratedImageToGalleryItem(image),
          preserveNavigationQuery,
          projectId,
          selectionPage,
          type: 'selectGalleryItem',
        }),
      /** Select an image found outside the Gallery; Preview navigates within the image's own board. */
      selectImageInItsBoard: (image: GeneratedImageContract & Partial<GalleryImage>, projectId?: string): void =>
        dispatch({
          item: legacyGeneratedImageToGalleryItem(image),
          navigateItemBoard: true,
          projectId,
          type: 'selectGalleryItem',
        }),
      setCompareImage: (image: (GeneratedImageContract & Partial<GalleryImage>) | null, projectId?: string): void =>
        dispatch({
          image: image ? legacyGeneratedImageToGalleryItem(image) : null,
          projectId,
          type: 'setGalleryCompareImage',
        }),
      selectBoard: command('selectGalleryBoard', (boardId: string, projectId?: string) => ({ boardId, projectId })),
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
      setStarredOnly: command('setGalleryStarredOnly', (starredOnly: boolean, projectId?: string) => ({
        projectId,
        starredOnly,
      })),
      setSemanticSearchMode: command('setGallerySemanticSearchMode', (enabled: boolean, projectId?: string) => ({
        enabled,
        projectId,
      })),
      setSemanticSearchText: command('setGallerySemanticSearchText', (text: string, projectId?: string) => ({
        projectId,
        text,
      })),
      commitSemanticSearch: command('commitGallerySemanticSearch', (text: string, projectId?: string) => ({
        projectId,
        text,
      })),
      clearSearch: command('clearGallerySearch', (projectId?: string) => ({ projectId })),
      setView: command(
        'setGalleryView',
        (galleryView: ActionPayload<'setGalleryView'>['galleryView'], projectId?: string) => ({
          galleryView,
          projectId,
        })
      ),
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
        (
          values: ActionPayload<'patchProjectPromptDraft'>['values'],
          sourceId: ActionPayload<'patchProjectPromptDraft'>['sourceId'],
          projectId?: string,
          origin?: ActionPayload<'patchProjectPromptDraft'>['origin']
        ) => ({ origin, projectId, sourceId, values })
      ),
      patchSettings: command(
        'patchGenerateSettings',
        (
          values: ActionPayload<'patchGenerateSettings'>['values'],
          projectId?: string,
          origin?: ActionPayload<'patchGenerateSettings'>['origin']
        ) => ({ origin, projectId, values })
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
        (
          values: ActionPayload<'setGenerateSettings'>['values'],
          projectId?: string,
          origin?: ActionPayload<'setGenerateSettings'>['origin']
        ) => ({ origin, projectId, values })
      ),
      setSource: command('setInvocationSource', (sourceId: ActionPayload<'setInvocationSource'>['sourceId']) => ({
        sourceId,
      })),
      submitCanvas: command('submitCanvasInvocationSnapshot'),
      submitResolved: command('submitResolvedInvocationSnapshot'),
      toggleDestinationLock: command('toggleDestinationLock'),
      toggleRoutingLock: command('toggleRoutingLock'),
      toggleSourceLock: command('toggleSourceLock'),
    },
    layout: {
      activatePreset: (presetId: ActionPayload<'applyPreset'>['presetId']) =>
        layoutPresetActivator.activate(resolveSavedLayoutPreset(getState().account, presetId)),
      /**
       * Drop an activation still waiting on its widgets, leaving the active preset as it is. Choosing the active preset
       * while another is pending means "stay", not the active preset's revert.
       */
      cancelPresetActivation: (): void => layoutPresetActivator.invalidate(),
      applyPreset: command('applyPreset', (presetId: ActionPayload<'applyPreset'>['presetId']) => ({ presetId })),
      createPreset: command(
        'addLayoutPreset',
        (
          presetId: ActionPayload<'addLayoutPreset'>['presetId'],
          label: string,
          iconId?: string,
          defaultRoute?: ActionPayload<'addLayoutPreset'>['defaultRoute']
        ) => ({
          defaultRoute,
          iconId,
          label,
          presetId,
        })
      ),
      reorderPresets: command(
        'reorderLayoutPresets',
        (
          activeId: ActionPayload<'reorderLayoutPresets'>['activeId'],
          overId: ActionPayload<'reorderLayoutPresets'>['overId']
        ) => ({ activeId, overId })
      ),
      setPresetIcon: command(
        'setLayoutPresetIcon',
        (presetId: ActionPayload<'setLayoutPresetIcon'>['presetId'], iconId: string) => ({ iconId, presetId })
      ),
      setPresetRoute: command(
        'setLayoutPresetRoute',
        (
          presetId: ActionPayload<'setLayoutPresetRoute'>['presetId'],
          defaultRoute: ActionPayload<'setLayoutPresetRoute'>['defaultRoute']
        ) => ({ defaultRoute, presetId })
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
      /** Discards the named preset's unsaved arrangement in the active project. */
      revertPreset: command('revertLayoutPreset', (presetId: ActionPayload<'revertLayoutPreset'>['presetId']) => ({
        presetId,
      })),
      /** Writes the preset's arrangement (live when active, else this project's working copy) back onto it. */
      savePreset: command('saveLayoutPreset', (presetId: ActionPayload<'saveLayoutPreset'>['presetId']) => ({
        presetId,
      })),
      /** Drops saved arrangement, identity, and route edits from a built-in preset. */
      restorePresetDefault: command(
        'restoreLayoutPresetDefault',
        (presetId: ActionPayload<'restoreLayoutPresetDefault'>['presetId']) => ({ presetId })
      ),
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
    // Commands that change the active or open projects first commit drafts still held by mounted editors, so an edit
    // made while a caller awaited (hydration, a close flush) lands on the project it was typed in.
    projects: {
      /** With `unchangedFrom`, closes only while the project, drafts included, is still exactly that version. */
      close: (projectId: string, unchangedFrom?: Project): ProjectCommandResult => {
        flushWorkbenchDrafts();
        const state = getState();
        const project = state.projects.find((project) => project.id === projectId);
        if (!project) {
          return { ok: false, reason: 'project-not-found' };
        }

        if (hasActiveQueueRuns(project)) {
          return { ok: false, reason: 'active-queue-runs' };
        }

        if (unchangedFrom && project !== unchangedFrom) {
          return { ok: false, reason: 'modified' };
        }

        if (state.projects.length === 1) {
          return { ok: false, reason: 'last-project' };
        }

        dispatch({ projectId, type: 'closeProject' });
        return { ok: true };
      },
      create: (): Project => {
        flushWorkbenchDrafts();
        dispatch({ type: 'createProject' });
        return getActiveProject(getState());
      },
      open: (project: Project): void => {
        flushWorkbenchDrafts();
        dispatch({ project, type: 'openProject' });
      },
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

        flushWorkbenchDrafts();
        dispatch({ projectId, type: 'switchProject' });
        return { ok: true };
      },
    },
    queue: {
      cancel: command(
        'cancelQueueItem',
        (projectId: string | undefined, queueItemId: string, notice?: ActionPayload<'cancelQueueItem'>['notice']) => ({
          projectId,
          queueItemId,
          ...(notice ? { notice } : {}),
        })
      ),
      cancelAll: command('cancelAllQueueItems', (projectId?: string) => ({ projectId })),
      cancelAllExceptCurrent: command(
        'cancelAllQueueItemsExceptCurrent',
        (projectId?: string, currentQueueItemId?: string | null) => ({ currentQueueItemId, projectId })
      ),
      clearCompleted: command('clearCompletedQueueItems'),
      markBackendCancelled: command('markQueueItemBackendCancelled'),
      markBackendSubmitted: command('markQueueItemBackendSubmitted'),
      setCancellationPending: command('setQueueItemCancellationPending'),
      setLocalRecoveryState: command('setQueueItemLocalRecoveryState'),
      routePartialResults: command('routeQueueItemPartialResults'),
      routeResults: command('routeQueueItemResults'),
      restoreFromJournal: command('restoreQueueItemsFromJournal'),
      setConnectionStatus: command('setBackendConnectionStatus'),
      setStatus: command('setQueueItemStatus'),
    },
    widgets: {
      dockFloating: command('dockFloatingWidget', (instanceId: string) => ({ instanceId })),
      closeFloating: command('closeFloatingWidget', (instanceId: string) => ({ instanceId })),
      float: command(
        'floatWidget',
        (
          instanceId: string,
          region?: ActionPayload<'floatWidget'>['region'],
          viewport?: ActionPayload<'floatWidget'>['viewport']
        ) => ({ instanceId, ...(region ? { region } : {}), ...(viewport ? { viewport } : {}) })
      ),
      raiseFloating: command('raiseFloatingWidget', (instanceId: string) => ({ instanceId })),
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
      patchValues: (
        widgetId: ActionPayload<'patchWidgetValues'>['widgetId'],
        values: Record<string, unknown> | ((current: Record<string, unknown>) => Record<string, unknown>),
        projectId?: string,
        origin?: ActionPayload<'patchWidgetValues'>['origin']
      ): void => {
        const state = getState();
        const project = state.projects.find((candidate) => candidate.id === (projectId ?? state.activeProjectId));

        if (!project) {
          return;
        }

        // Resolve before dispatch so same-turn draft commits compose, without storing callbacks in reducer actions.
        dispatch({
          origin,
          projectId: project.id,
          type: 'patchWidgetValues',
          values: typeof values === 'function' ? values(getProjectWidgetValues(project, widgetId)) : values,
          widgetId,
        });
      },
      reorder: command('reorderWidgetInstances'),
      revealFloating: command('revealFloatingWidget', (instanceId: string) => ({ instanceId })),
      setAlignment: command('setWidgetInstanceAlignment'),
      select: command('selectRegionWidget'),
      setFloatingGeometry: command(
        'setFloatingWidgetGeometry',
        (instanceId: string, geometry: { x: number; y: number; widthPx: number; heightPx: number }) => ({
          instanceId,
          ...geometry,
        })
      ),
      setFloatingMode: command(
        'setFloatingWidgetMode',
        (instanceId: string, mode: ActionPayload<'setFloatingWidgetMode'>['mode']) => ({ instanceId, mode })
      ),
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
      /** Adds a workflow to the project and activates it; returns the id it is known by. */
      add: (
        document: ProjectGraphState,
        options: { label: string; projectId?: string; reusePlaceholder?: boolean; source?: ProjectWorkflowSource }
      ): string => {
        dispatch({
          document,
          label: options.label,
          projectId: options.projectId,
          reusePlaceholder: options.reusePlaceholder,
          source: options.source,
          type: 'addProjectWorkflow',
        });
        return document.id;
      },
      create: (projectId?: string): string => {
        const document = createBlankWorkflowDocument();
        dispatch({ document, label: 'New workflow', projectId, type: 'addProjectWorkflow' });
        return document.id;
      },
      duplicate: (workflowId: string, copyName: string, projectId?: string): string | null => {
        const project = getState().projects.find(
          (candidate) => candidate.id === (projectId ?? getState().activeProjectId)
        );
        if (!project || !findProjectWorkflow(project, workflowId)) {
          return null;
        }
        const copyId = createBlankWorkflowDocument().id;
        dispatch({ copyId, copyName, projectId, type: 'duplicateProjectWorkflow', workflowId });
        return copyId;
      },
      editGraph: command(
        'applyWorkflowAction',
        (action: ProjectGraphAction, target?: { projectId: string; workflowId: string }) => ({
          action,
          projectId: target?.projectId,
          workflowId: target?.workflowId,
        })
      ),
      redo: command('redoWorkflowChange', (target?: { projectId?: string; workflowId?: string }) => ({
        projectId: target?.projectId,
        workflowId: target?.workflowId,
      })),
      remove: command('removeProjectWorkflow', (workflowId: string, projectId?: string) => ({ projectId, workflowId })),
      rename: (workflowId: string, name: string, projectId?: string): ProjectCommandResult => {
        if (!name.trim()) {
          return { ok: false, reason: 'invalid-name' };
        }
        dispatch({
          action: { name: name.trim(), type: 'renameWorkflow' },
          projectId,
          type: 'applyWorkflowAction',
          workflowId,
        });
        return { ok: true };
      },
      /** Replaces a copy's document with a fresh load and activates it; one undo step restores the old one. */
      replaceDocument: command(
        'replaceProjectWorkflowDocument',
        (
          target: { projectId: string; workflowId: string },
          document: ProjectGraphState,
          options: { label: string; source: ProjectWorkflowSource }
        ) => ({ ...target, ...options, document })
      ),
      select: command('selectProjectWorkflow', (workflowId: string, projectId?: string) => ({ projectId, workflowId })),
      /** Records where a workflow was explicitly published; a no-op once the project or workflow is gone. */
      setSource: command(
        'setProjectWorkflowSource',
        (projectId: string, workflowId: string, source: ProjectWorkflowSource | undefined) => ({
          projectId,
          source,
          workflowId,
        })
      ),
      undo: command('undoWorkflowChange', (target?: { projectId?: string; workflowId?: string }) => ({
        projectId: target?.projectId,
        workflowId: target?.workflowId,
      })),
    },
  };
};

const createPersistenceAdapter = (dispatch: WorkbenchDispatch, getState: () => WorkbenchState) => {
  const command = createCommandFactory(dispatch);

  return {
    /**
     * Record the server-assigned board without selecting it: the user may have changed selection during creation,
     * and unresolved selections already fall back to the project board.
     */
    assignProjectBoard: ({ boardId, projectId }: { boardId: string; projectId: string }) => {
      dispatch({ boardId, projectId, type: 'setGalleryProjectBoardId' });
    },
    getState,
    hydrate: command('hydrateWorkbench', (state: WorkbenchState) => ({ state })),
    replaceProjectFromServer: command('replaceProjectFromServer'),
    retargetProject: (payload: ActionPayload<'retargetProject'>): ProjectCommandResult => {
      const state = getState();
      if (!state.projects.some((project) => project.id === payload.projectId)) {
        return { ok: false, reason: 'project-not-found' };
      }
      const source = state.projects.find((project) => project.id === payload.projectId);
      if (source && hasInFlightQueueRuns(source)) {
        return { ok: false, reason: 'active-queue-runs' };
      }
      if (state.projects.some((project) => project.id === payload.targetProjectId)) {
        return { ok: false, reason: 'target-already-open' };
      }
      dispatch({ ...payload, type: 'retargetProject' });
      return { ok: true };
    },
    saveFailed: command('autosaveFailed', (error: string) => ({ error })),
    savePending: command('autosavePending', (error: string) => ({ error })),
    saveScheduled: command('autosaveScheduled'),
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

export interface WorkbenchStoreOptions {
  isLoaded?: (preset: LayoutPreset) => boolean;
  loadLayoutPresetWidgets?: (preset: LayoutPreset) => Promise<unknown>;
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

/** The store is the one diagnostic owner for reported errors, widget failures and autosave outcomes. */
const recordDiagnosticForAction = (
  action: WorkbenchAction,
  previousState: WorkbenchState,
  nextState: WorkbenchState
): void => {
  switch (action.type) {
    case 'recordError': {
      const { error, ...context } = action.context ?? {};

      recordLogEvent(
        'error',
        {
          area: action.area ?? 'runtime',
          namespace: action.namespace ?? 'system',
          projectId: getDiagnosticProjectId(nextState, action.projectId),
        },
        {
          context: Object.keys(context).length > 0 ? context : undefined,
          error,
          message: action.message,
          name: `${action.namespace ?? 'system'}.${action.area ?? 'runtime'}`,
        }
      );
      break;
    }
    case 'recordWidgetFailure': {
      if (Object.is(previousState, nextState)) {
        return;
      }

      recordLogEvent(
        'error',
        { area: 'widget-failure', namespace: 'system', projectId: getDiagnosticProjectId(nextState) },
        {
          context: { details: action.failure.details, widgetId: action.failure.widgetId },
          message: action.failure.message,
          name: 'widget.registration-failed',
        }
      );
      break;
    }
    case 'closeProject': {
      if (previousState.projects.length !== 1) {
        return;
      }

      recordLogEvent(
        'error',
        {
          area: 'project-lifecycle',
          namespace: 'system',
          projectId: getDiagnosticProjectId(nextState, action.projectId),
        },
        { message: 'At least one project must remain open.', name: 'project.close-refused' }
      );
      break;
    }
    case 'markQueueItemBackendSubmitted': {
      recordLogEvent(
        'info',
        { area: 'submission', namespace: 'queue', projectId: action.projectId },
        {
          context: {
            backendBatchId: action.backendBatchId,
            backendItemIds: action.backendItemIds,
            queueItemId: action.queueItemId,
          },
          message: 'Queue item accepted by the backend',
          name: 'queue.submitted',
        }
      );
      break;
    }
    case 'setQueueItemStatus': {
      if (Object.is(previousState, nextState)) {
        return;
      }

      const level = action.status === 'failed' ? 'error' : 'debug';

      recordLogEvent(
        level,
        { area: 'history', namespace: 'queue', projectId: action.projectId },
        {
          context: { queueItemId: action.queueItemId, reason: action.error, status: action.status },
          message:
            action.status === 'failed'
              ? `Queue item failed${action.error ? `: ${action.error}` : ''}`
              : `Queue item ${action.status}`,
          name: action.status === 'failed' ? 'queue.item-failed' : 'queue.item-status',
        }
      );
      break;
    }
    case 'hydrateWorkbench': {
      recordLogEvent(
        'info',
        { area: 'hydration', namespace: 'persistence' },
        {
          context: { activeProjectId: nextState.activeProjectId, projectCount: nextState.projects.length },
          message: 'Workbench hydrated',
          name: 'persistence.hydrated',
        }
      );
      break;
    }
    case 'autosaveStarted': {
      recordLogEvent(
        'debug',
        { area: 'autosave', namespace: 'persistence', projectId: nextState.activeProjectId },
        { message: 'Autosave started', name: 'persistence.autosave-started' }
      );
      break;
    }
    case 'autosaveSucceeded': {
      recordLogEvent(
        'debug',
        { area: 'autosave', namespace: 'persistence', projectId: nextState.activeProjectId },
        { context: { savedAt: action.savedAt }, message: 'Autosave succeeded', name: 'persistence.autosave-succeeded' }
      );
      break;
    }
    case 'autosavePending': {
      recordLogEvent(
        'warn',
        { area: 'autosave', namespace: 'persistence', projectId: nextState.activeProjectId },
        { context: { reason: action.error }, message: 'Autosave needs attention', name: 'persistence.autosave-pending' }
      );
      break;
    }
    case 'autosaveFailed': {
      recordLogEvent(
        'error',
        { area: 'autosave', namespace: 'persistence', projectId: nextState.activeProjectId },
        { context: { reason: action.error }, message: 'Autosave failed', name: 'persistence.autosave-failed' }
      );
      break;
    }
  }
};

export const createWorkbenchStore = (
  initialState = createInitialWorkbenchState(),
  options: WorkbenchStoreOptions = {}
): WorkbenchInternalStore => {
  let state = initialState;
  let hasHydrated = false;
  let persistedRevision = 0;
  let invalidateLayoutPresetActivation = (): void => undefined;
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
    const nextState = __workbenchReducerInternal(state, action, {
      autoSwitchInvocationRoute: getWorkbenchPreferences().autoSwitchInvocationRoute,
    });

    if (nextState !== previousState && visibleWidgetsKey(nextState) !== visibleWidgetsKey(previousState)) {
      closeWidgetOverlays();
    }

    if (
      action.type === 'applyPreset' ||
      action.type === 'hydrateWorkbench' ||
      previousState.activeProjectId !== nextState.activeProjectId
    ) {
      invalidateLayoutPresetActivation();
    }

    if (action.type === 'hydrateWorkbench') {
      clearLayerPanelStates();
    } else if (nextState !== previousState) {
      // The always-live store boundary collapses stale secondaries after an
      // external primary change or layer removal, even while the Layers widget
      // is unmounted.
      reconcileLayerPanelStates(nextState.projects);
    }

    setSnapshotState(nextState);
    recordDiagnosticForAction(action, previousState, nextState);
  };

  const getState = (): WorkbenchState => state;
  const layoutPresetActivator = createLayoutPresetActivator({
    apply: (presetId) => dispatch({ presetId, type: 'applyPreset' }),
    getActiveProjectId: () => state.activeProjectId,
    isCurrent: (preset) => {
      const current = resolveSavedLayoutPreset(state.account, preset.id);

      return current.id === preset.id && current.snapshot === preset.snapshot;
    },
    // Switching lays out the active project's working copy when it has one, so that is what has to be in memory.
    isLoaded: (preset) => {
      const incoming = { ...preset, snapshot: getLayoutPresetArrangement(getActiveProject(state), preset) };

      return options.isLoaded
        ? options.isLoaded(incoming)
        : areWidgetsLoaded(getLayoutWidgetTypeIds(incoming.snapshot));
    },
    load: (preset) =>
      (options.loadLayoutPresetWidgets ?? loadLayoutPresetWidgets)({
        ...preset,
        snapshot: getLayoutPresetArrangement(getActiveProject(state), preset),
      }),
  });
  invalidateLayoutPresetActivation = layoutPresetActivator.invalidate;
  const commands = createCommands(dispatch, getState, layoutPresetActivator);

  return {
    commands,
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
