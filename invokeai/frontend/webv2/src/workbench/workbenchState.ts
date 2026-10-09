import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { QueueCompiledSubmission, QueueHistoryItemStatus } from '@features/queue/contracts';
import type { ProjectGraphState, ProjectWorkflowSource } from '@features/workflow/contracts';
import type { WorkflowSubmissionPlan } from '@features/workflow/graph';
import type { LogNamespace } from '@platform/logging/contracts';
import type {
  CanvasDocumentContractV3,
  CanvasPlacementContract,
  CanvasStateContractV3,
  CanvasStagingCandidateContract,
} from '@workbench/canvas-engine/api';
import type { GraphContract } from '@workbench/graphContracts';
import type {
  ExpandedPositivePrompts,
  InvocationRoute,
  InvocationSourceId,
  ResultDestination,
} from '@workbench/invocationContracts';
import type {
  BuiltInLayoutPresetId,
  CenterViewId,
  FloatingWidgetMode,
  FloatingWidgetState,
  LayoutPreset,
  LayoutPresetId,
  LayoutPresetMetadataOverride,
  LayoutPresetMetadataOverrides,
  LayoutPresetOverrides,
  LayoutPresetRoute,
  LayoutPresetRouteOverrides,
  LayoutPresetSnapshot,
  LayoutPresetWorkingCopy,
  ProjectLayoutState,
  WidgetRegion,
  WidgetRegionState,
} from '@workbench/layoutContracts';
import type {
  AccountState,
  Project,
  ProjectUndoSnapshot,
  PromptHistoryItem,
  WorkbenchNotification,
  WorkbenchNotificationCategory,
  ProjectLoadResult,
  WorkbenchNotificationKind,
  WorkbenchState,
} from '@workbench/projectContracts';
import type { ProjectSettings } from '@workbench/settings/contracts';
import type {
  WidgetFailure,
  WidgetId,
  WidgetInstanceContract,
  WidgetInstanceId,
  WidgetStateContract,
  WidgetStateMap,
  WidgetTypeId,
} from '@workbench/widgetContracts';

import {
  getBoundedRecentImages,
  getPersistedSelectedGalleryItemKeys,
  stripInfiniteWindowAnchor,
  stripUnresolvableGallerySearch,
  GALLERY_AUTO_ADD_FOLLOW,
  gallerySemanticReferenceKey,
  getGallerySettings,
  parseGallerySemanticReference,
  toGallerySemanticTextReference,
  getGalleryAutoAddBoardId,
  getGallerySelectionBoardId,
  getSelectedGalleryItemFromValues,
  legacyGeneratedImageToGalleryItem,
  normalizeGalleryImage,
  parseGalleryItemKey,
  toGalleryItemKey,
  type GalleryImage,
  type GalleryImageItem,
  type GalleryItem,
  type GalleryItemKey,
  type GalleryBoardDeletionResult,
  type GallerySettings,
  type GeneratedImageContract,
  type GeneratedVideoContract,
  generatedVideoToGalleryItem,
} from '@features/gallery/contracts';
import { planSeedSubmission } from '@platform/core/seed';
import { describeError } from '@platform/logging/normalize';
import { WIDGET_REGIONS } from '@workbench/layoutContracts';
import { prependProjectEvent, PROJECT_EVENT_LIMIT } from '@workbench/projectEvents';

import type { WorkbenchQueueItem as QueueItem } from './queueHistoryContracts';

import {
  getChangedValueKeys,
  getRouteAfterHighConfidenceEdit,
  isHighConfidenceCanvasEdit,
  isHighConfidenceCanvasEditIntent,
  isHighConfidenceGenerateEdit,
  isHighConfidenceGraphEdit,
  isHighConfidenceUpscaleEdit,
  isHighConfidenceVideoEdit,
  type CanvasEditIntent,
  type WorkbenchActionOrigin,
} from './autoRoutePolicy';
import { createNewCanvasState, loadCanvasState } from './canvasMigration';
import { applyCanvasProjectMutation, type CanvasProjectMutation } from './canvasProjectMutations';
import { gateProjectCanvases } from './projectCanvasGate';
import {
  addProjectWorkflow,
  applyProjectWorkflowAction,
  createBlankWorkflowDocument,
  createProjectWorkflowCollection,
  duplicateProjectWorkflow,
  findProjectWorkflow,
  getActiveProjectGraph,
  getActiveProjectWorkflow,
  migrateProjectGraphToCollection,
  normalizeProjectWorkflowCollection,
  recordProjectWorkflowRun,
  redoProjectWorkflow,
  removeProjectWorkflow,
  replaceProjectWorkflowDocument,
  selectProjectWorkflow,
  setProjectWorkflowDocument,
  setProjectWorkflowSource,
  undoProjectWorkflow,
  type ProjectWorkflowCollection,
} from './projectWorkflows';
import { normalizeRestoredQueueItem } from './queue-integration/queueRunRestoration';
import { getProjectWidgetValues } from './widgetState';
export { nextLayerName } from './canvasProjectMutations';
import type { ProjectPromptDraftPatch } from '@features/generation/settings';

import { compileGenerateGraph, resolveGenerateSeed } from '@features/generation/graph';
import {
  addPromptHistoryItem,
  applyProjectPromptDraft,
  cloneGenerateWidgetValues,
  createDynamicPromptsSampleSeed,
  getEffectivePrompts,
  getGenerationModelAvailabilityReasons,
  getPromptDraftFromValues,
  getPromptHistoryItemFromGenerateSettings,
  hasDynamicPromptSyntax,
  migrateProjectPromptDraft,
  normalizeGenerateSettings,
  normalizeGenerateWidgetValues,
  removePromptHistoryItem,
  sanitizeBatchCount,
  syncGenerateWidgetValuesWithModels,
} from '@features/generation/settings';
import { MAX_QUEUE_BATCH_ITEMS } from '@features/queue';
import { getGenerationDevicesSnapshot, resolveRandDeviceMetadata } from '@features/queue/devices';
import {
  clearDeletedUpscaleInput,
  cloneUpscaleWidgetValues,
  compileUpscaleGraph,
  getUpscaleOutputDimensions,
  getUpscaleValidationReasons,
  normalizeUpscaleWidgetValues,
  resolveUpscaleSeed,
  syncUpscaleWidgetValuesWithModels,
  type UpscaleWidgetValues,
} from '@features/upscale';
import {
  clearDeletedVideoMedia,
  cloneVideoWidgetValues,
  compileVideoGraph,
  getVideoDimensions,
  getVideoWidgetValidationReasons,
  normalizeVideoWidgetValues,
  resolveVideoSeed,
  syncVideoWidgetValuesWithModels,
  type VideoWidgetValues,
} from '@features/video';
import { planWorkflowSubmission } from '@features/workflow/graph';
import { getInvocationTemplatesSnapshot } from '@features/workflow/react';
import {
  hasMultipleWorkflowReturnNodes,
  projectGraphReducer,
  serializeWorkflowJsonForSubmission,
  type ProjectGraphAction,
  type WorkflowGeneratorResolutions,
} from '@features/workflow/utility';

import {
  getCanvasStagingSlotCount,
  getCanvasStagingSlots,
  getFirstCanvasPlaceholderSlotIndex,
  type CanvasStagingSlot,
  isCancellableQueueItem,
} from './canvasStagingView';
import {
  cascadeDefaultGeometry,
  clampSizeToMinimum,
  fitWindowIntoViewport,
  getRegionOrder,
  isAwaitedCenterView,
  nextStackOrder,
  normalizeFloatingPlacement,
  normalizeLastFloatingGeometry,
  rememberClosedWindows,
  toFloatingWidgetState,
  withoutFloatedInstances,
  writeRegionOrder,
  type RegionOrderSlot,
} from './floatingWindows';
import { getSourceIdForWidgetTypeId } from './graphWidgets';
import {
  defaultInvocationRoute,
  isInvocationRouteValid,
  isInvocationSourceAvailable,
  isResultDestinationAvailable,
  resolveInvocationRoute,
} from './invocation';
import { getOrderedLayoutPresets, normalizeLayoutPresetOrder, reorderLayoutPresetIds } from './layoutPresetCollection';
import { getInvocationAfterLayoutPreset } from './layoutPresetRouting';
import {
  defaultLayoutPreset,
  getLayoutPreset,
  isBuiltInLayoutPresetId,
  layoutPresets,
  resolveLayoutPresetId,
} from './layoutPresets';
import {
  areLayoutPresetSnapshotsEqual,
  cloneLayoutPresetWidgetRegions,
  createLayoutPresetSnapshot,
  doesLayoutPresetExist,
  findLayoutPresetWorkingCopy,
  normalizeLayoutPresetSnapshot,
  resolveSavedLayoutPreset,
} from './layoutPresetSnapshots';
import { normalizeProjectSettings } from './settings/store';

interface QueueGenerateSnapshot {
  negativePromptNodeId: string;
  positivePromptNodeId: string;
  seedNodeId: string;
  values: GenerateWidgetValues;
}

export interface WorkbenchReducerContext {
  autoSwitchInvocationRoute: boolean;
}

type WorkbenchReducerAction =
  | { type: 'createProject' }
  | { type: 'openProject'; project: Project }
  | { type: 'closeProject'; projectId: string }
  | { type: 'renameProject'; projectId: string; name: string }
  | { type: 'switchProject'; projectId: string }
  | { type: 'setCenterView'; centerViewId: CenterViewId }
  | { type: 'applyPreset'; presetId: LayoutPresetId }
  | { type: 'reorderLayoutPresets'; activeId: LayoutPresetId; overId: LayoutPresetId }
  | {
      type: 'addLayoutPreset';
      presetId: LayoutPresetId;
      label: string;
      iconId?: string;
      defaultRoute?: LayoutPresetRoute | null;
    }
  | { type: 'setLayoutPresetIcon'; presetId: LayoutPresetId; iconId: string }
  | { type: 'setLayoutPresetRoute'; presetId: LayoutPresetId; defaultRoute: LayoutPresetRoute | null }
  | { type: 'saveLayoutPreset'; presetId: LayoutPresetId }
  | { type: 'restoreLayoutPresetDefault'; presetId: LayoutPresetId }
  | { type: 'renameLayoutPreset'; presetId: LayoutPresetId; label: string }
  | { type: 'deleteLayoutPreset'; presetId: LayoutPresetId }
  | { type: 'resetActiveLayout' }
  /** Drop a preset's unsaved arrangement: the active one reapplies its saved layout, an inactive one forgets its copy. */
  | { type: 'revertLayoutPreset'; presetId: LayoutPresetId }
  | { type: 'recoverShellLayout' }
  | { type: 'setInvocationSource'; sourceId: InvocationSourceId }
  | { type: 'setInvocationDestination'; destination: ResultDestination }
  | { type: 'toggleRoutingLock' }
  | { type: 'toggleSourceLock' }
  | { type: 'toggleDestinationLock' }
  | {
      type: 'openRegionWidget';
      region: WidgetRegion;
      widgetId: WidgetTypeId;
      createNew?: boolean;
      initialValues?: Record<string, unknown>;
      projectId?: string;
    }
  | { type: 'selectRegionWidget'; region: WidgetRegion; widgetId: WidgetInstanceId; projectId?: string }
  | { type: 'toggleRegionWidget'; region: WidgetRegion; widgetId: WidgetInstanceId; projectId?: string }
  | {
      type: 'moveWidgetInstance';
      instanceId: WidgetInstanceId;
      fromRegion: WidgetRegion;
      toRegion: WidgetRegion;
      toIndex: number;
    }
  | {
      type: 'reorderWidgetInstances';
      region: WidgetRegion;
      activeInstanceId?: WidgetInstanceId;
      instanceIds: WidgetInstanceId[];
    }
  | { type: 'setWidgetInstanceAlignment'; region: WidgetRegion; instanceId: WidgetInstanceId; align: 'start' | 'end' }
  | { type: 'setRegionWidgetCollapsed'; region: WidgetRegion; isCollapsed: boolean }
  | { type: 'setRegionWidgetSize'; region: WidgetRegion; sizePx: number }
  | {
      type: 'floatWidget';
      instanceId: WidgetInstanceId;
      /** The chrome the float was asked from; docking returns the window there. */
      region?: WidgetRegion;
      /** The viewport the window opens into, so a remembered position from a larger one is brought on screen. */
      viewport?: { width: number; height: number };
    }
  | { type: 'dockFloatingWidget'; instanceId: WidgetInstanceId }
  | { type: 'closeFloatingWidget'; instanceId: WidgetInstanceId }
  | {
      type: 'setFloatingWidgetGeometry';
      instanceId: WidgetInstanceId;
      x: number;
      y: number;
      widthPx: number;
      heightPx: number;
    }
  | { type: 'setFloatingWidgetMode'; instanceId: WidgetInstanceId; mode: FloatingWidgetMode }
  | { type: 'raiseFloatingWidget'; instanceId: WidgetInstanceId }
  /** Bring a window forward so its content shows: raise it, and expand it when it is shaded. */
  | { type: 'revealFloatingWidget'; instanceId: WidgetInstanceId }
  | { type: 'setGenerateSettings'; values: GenerateWidgetValues; projectId?: string; origin?: WorkbenchActionOrigin }
  | {
      type: 'patchGenerateSettings';
      values: Partial<GenerateWidgetValues>;
      projectId?: string;
      origin?: WorkbenchActionOrigin;
    }
  | {
      type: 'patchProjectPromptDraft';
      values: ProjectPromptDraftPatch;
      sourceId: 'generate' | 'upscale';
      projectId?: string;
      origin?: WorkbenchActionOrigin;
    }
  | { type: 'setGenerateBatchCount'; batchCount: number; projectId?: string }
  | { type: 'addPromptToHistory'; prompt: PromptHistoryItem; projectId?: string }
  | { type: 'removePromptFromHistory'; prompt: PromptHistoryItem; projectId?: string }
  | { type: 'clearPromptHistory'; projectId?: string }
  | {
      type: 'patchWidgetValues';
      widgetId: WidgetTypeId;
      values: Record<string, unknown>;
      projectId?: string;
      origin?: WorkbenchActionOrigin;
    }
  | {
      type: 'patchWidgetInstanceValues';
      instanceId: WidgetInstanceId;
      values: Record<string, unknown>;
      projectId?: string;
    }
  | {
      type: 'setWidgetInstanceValues';
      instanceId: WidgetInstanceId;
      values: Record<string, unknown>;
      projectId?: string;
    }
  /** Edits one workflow; absent targets mean the active project's active workflow. */
  | { type: 'applyWorkflowAction'; action: ProjectGraphAction; projectId?: string; workflowId?: string }
  | {
      type: 'addProjectWorkflow';
      document: ProjectGraphState;
      label: string;
      projectId?: string;
      source?: ProjectWorkflowSource;
      reusePlaceholder?: boolean;
    }
  | { type: 'selectProjectWorkflow'; workflowId: string; projectId?: string }
  | { type: 'duplicateProjectWorkflow'; workflowId: string; copyId: string; copyName: string; projectId?: string }
  | { type: 'removeProjectWorkflow'; workflowId: string; projectId?: string }
  | { type: 'setProjectWorkflowSource'; projectId: string; workflowId: string; source?: ProjectWorkflowSource }
  | {
      type: 'replaceProjectWorkflowDocument';
      projectId: string;
      workflowId: string;
      document: ProjectGraphState;
      label: string;
      source: ProjectWorkflowSource;
    }
  | { type: 'undoWorkflowChange'; projectId?: string; workflowId?: string }
  | { type: 'redoWorkflowChange'; projectId?: string; workflowId?: string }
  | { type: 'submitInvocationSnapshot'; backendSupportsCancellation: boolean; models?: readonly ModelConfig[] }
  | ({
      type: 'submitResolvedInvocationSnapshot';
      backendSupportsCancellation: boolean;
      /** Async workflow generator outputs, resolved by the caller before dispatch. */
      workflowGenerators?: WorkflowGeneratorResolutions;
      route: InvocationRoute;
      models?: readonly ModelConfig[];
      /** The project the submission was prepared from; a switch in between never redirects it. */
      projectId: string;
      /** The workflow the submission was prepared from; required for workflow routes. */
      workflowId?: string;
    } & Partial<ExpandedPositivePrompts>)
  | {
      type: 'markQueueItemBackendSubmitted';
      projectId: string;
      queueItemId: string;
      backendItemIds: number[];
      backendBatchId?: string;
    }
  | {
      type: 'setQueueItemStatus';
      projectId: string;
      queueItemId: string;
      status: QueueHistoryItemStatus;
      error?: string;
      notify?: boolean;
    }
  | {
      type: 'routeQueueItemPartialResults';
      projectId: string;
      queueItemId: string;
      backendItemId: number;
      images: GeneratedImageContract[];
      videos?: GeneratedVideoContract[];
    }
  | { type: 'markQueueItemBackendCancelled'; projectId: string; queueItemId: string; backendItemId: number }
  | { type: 'setQueueItemCancellationPending'; projectId: string; queueItemId: string; pending: boolean }
  | {
      type: 'setQueueItemLocalRecoveryState';
      projectId: string;
      queueItemId: string;
      state: NonNullable<QueueItem['localRecoveryState']>;
    }
  | {
      type: 'routeQueueItemResults';
      projectId: string;
      queueItemId: string;
      images: GeneratedImageContract[];
      videos?: GeneratedVideoContract[];
    }
  | { type: 'restoreQueueItemsFromJournal'; projectId: string; items: unknown[] }
  | { type: 'appendCanvasStagingCandidate'; projectId: string; candidate: CanvasStagingCandidateContract }
  | {
      type: 'selectGalleryItem';
      item: GalleryItem;
      preserveNavigationQuery?: boolean;
      projectId?: string;
      selectionPage?: number;
    }
  | {
      type: 'toggleGalleryItemInSelection';
      item: GalleryItem;
      nextPrimaryItem: GalleryItem | null;
      projectId?: string;
    }
  | {
      type: 'setGalleryMultiSelection';
      itemKeys: GalleryItemKey[];
      primaryItem: GalleryItem;
      /** Keep the navigation query already on the selection, as `selectGalleryItem` does for a host's own window. */
      preserveNavigationQuery?: boolean;
      projectId?: string;
      /** Stamps this page instead of the grid's. */
      selectionPage?: number;
    }
  | { type: 'setGalleryCompareImage'; image: GalleryImageItem | null; projectId?: string }
  | { type: 'selectGalleryBoard'; boardId: string; projectId?: string }
  | { type: 'clearGallerySelection'; projectId?: string }
  | { type: 'setGalleryView'; galleryView: 'images' | 'assets'; projectId?: string }
  | { type: 'setGallerySearchTerm'; searchTerm: string; projectId?: string }
  /** Toggles the search field between metadata search and semantic search, carrying its text across. */
  | { type: 'setGallerySemanticSearchMode'; enabled: boolean; projectId?: string }
  /** The semantic field's live text; the ranking follows only on commit. */
  | { type: 'setGallerySemanticSearchText'; text: string; projectId?: string }
  /** Applies the semantic text as the ranking, if it is still what the field holds. */
  | { type: 'commitGallerySemanticSearch'; text: string; projectId?: string }
  /** The field's clear button: drops the text, the ranking, and semantic mode together. */
  | { type: 'clearGallerySearch'; projectId?: string }
  | { type: 'setGalleryStarredOnly'; starredOnly: boolean; projectId?: string }
  | { type: 'updateGallerySettings'; settings: Partial<GallerySettings>; projectId?: string }
  | { type: 'setGalleryPage'; page: number; projectId?: string }
  | { type: 'setGalleryPageInfo'; totalImages: number; projectId?: string }
  | {
      type: 'patchGalleryItems';
      changes: Partial<Pick<GalleryItem, 'boardId' | 'starred'>>;
      itemKeys: GalleryItemKey[];
    }
  | { type: 'removeGalleryItems'; itemKeys: GalleryItemKey[] }
  | {
      type: 'reconcileDeletedGalleryBoard';
      outcome: GalleryBoardDeletionResult;
    }
  | { type: 'setGalleryProjectBoardId'; boardId: string; projectId?: string }
  | {
      type: 'applyCanvasProjectMutation';
      projectId: string;
      mutation: CanvasProjectMutation;
      origin?: WorkbenchActionOrigin;
    }
  | { type: 'commitCanvasEdit'; projectId: string; intent: CanvasEditIntent }
  | ({
      type: 'submitCanvasInvocationSnapshot';
      backendSupportsCancellation: boolean;
      canvas: CanvasStateContractV3;
      destination: ResultDestination;
      generate: QueueGenerateSnapshot;
      graph: GraphContract;
      projectId: string;
    } & Partial<ExpandedPositivePrompts>)
  | {
      type: 'cancelQueueItem';
      queueItemId: string;
      projectId?: string;
      /** Replaces the generic cancellation notice when the caller knows why the run stopped. */
      notice?: { title: string; message: string };
    }
  | { type: 'cancelAllQueueItems'; projectId?: string }
  | { type: 'cancelAllQueueItemsExceptCurrent'; projectId?: string; currentQueueItemId?: string | null }
  | { type: 'clearCompletedQueueItems' }
  | { type: 'undoProjectChange' }
  | { type: 'redoProjectChange' }
  | { type: 'hydrateWorkbench'; state: WorkbenchState }
  | { type: 'replaceProjectFromServer'; projectId: string; project: Project }
  | {
      type: 'retargetProject';
      boardId: string;
      name: string;
      project: Project;
      projectId: string;
      sourceName: string;
      targetProjectId: string;
    }
  | { type: 'autosaveScheduled' }
  | { type: 'autosaveStarted' }
  | { type: 'autosavePending'; error: string }
  | { type: 'autosaveSucceeded'; savedAt: string }
  | { type: 'autosaveFailed'; error: string }
  | { type: 'markAllNotificationsRead' }
  | { type: 'clearNotifications' }
  | { type: 'recordWidgetFailure'; failure: WidgetFailure }
  | { type: 'setActiveProjectSettings'; settings: Partial<ProjectSettings> }
  | {
      type: 'recordError';
      message: string;
      area?: string;
      /** `error` may be a raw Error; notifications show its message and diagnostics keep its stack. */
      context?: { error?: unknown; [key: string]: unknown };
      namespace?: LogNamespace;
      projectId?: string;
    }
  | { type: 'setBackendConnectionStatus'; status: WorkbenchState['backendConnection']['status']; error?: string }
  | { type: 'recordNotice'; kind: WorkbenchNotificationKind; title: string; message?: string };

const HISTORY_LIMIT = 40;
/** A pause this long between same-key edits (typing, dragging) starts a new undo step. */
const UNDO_MERGE_WINDOW_MS = 1500;
const NOTIFICATION_LIMIT = 100;
// Side-panel bounds keep widget controls usable while reserving work-surface space. The bottom status strip has
// separate bounds.
const MIN_PANEL_SIZE_PX = 350;
const MAX_PANEL_SIZE_PX = 720;
const MIN_STATUS_PANEL_SIZE_PX = 96;
const MAX_STATUS_PANEL_SIZE_PX = 420;

/** Overshoot required to collapse rather than stop at the minimum size. */
const PANEL_COLLAPSE_OVERSHOOT_PX = 80;

/** The resize bounds for a widget region — shared with the resize handles. */
export const getPanelSizeBounds = (region: WidgetRegion): { max: number; min: number } => {
  if (region === 'bottom') {
    return { max: MAX_STATUS_PANEL_SIZE_PX, min: MIN_STATUS_PANEL_SIZE_PX };
  }

  return { max: MAX_PANEL_SIZE_PX, min: MIN_PANEL_SIZE_PX };
};

/** The size at or below which a resize drag snaps the region shut. */
export const getPanelCollapseThreshold = (region: WidgetRegion): number =>
  getPanelSizeBounds(region).min - PANEL_COLLAPSE_OVERSHOOT_PX;

/** The collapse threshold for a panel currently rendered at `visibleSizePx`. */
export const getVisiblePanelCollapseThreshold = (region: WidgetRegion, visibleSizePx: number): number =>
  Math.min(getPanelCollapseThreshold(region), visibleSizePx - PANEL_COLLAPSE_OVERSHOOT_PX);

const now = (): string => new Date().toISOString();

const createId = (prefix: string): string =>
  `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

const createNotification = ({
  category,
  kind,
  message,
  messageKey,
  projectId,
  title,
  titleKey,
}: {
  category?: WorkbenchNotificationCategory;
  kind: WorkbenchNotificationKind;
  message?: string;
  messageKey?: string;
  projectId?: string;
  title: string;
  titleKey?: string;
}): WorkbenchNotification => ({
  category,
  createdAt: now(),
  id: createId('notification'),
  isRead: false,
  kind,
  message,
  messageKey,
  projectId,
  title,
  titleKey,
});

const addNotification = (state: WorkbenchState, notification: WorkbenchNotification): WorkbenchState => {
  const [newest, ...rest] = state.notifications;

  // Reuse the newest matching error's id to suppress repeated toasts. Routine success notifications must still
  // toast independently.
  if (
    newest &&
    newest.kind === 'error' &&
    newest.kind === notification.kind &&
    newest.title === notification.title &&
    newest.message === notification.message
  ) {
    return {
      ...state,
      notifications: [
        {
          ...newest,
          createdAt: notification.createdAt,
          isRead: false,
          occurrenceCount: (newest.occurrenceCount ?? 1) + 1,
        },
        ...rest,
      ],
    };
  }

  return { ...state, notifications: [notification, ...state.notifications].slice(0, NOTIFICATION_LIMIT) };
};

/** Adds queue feedback iff the reduction actually grew that project's queue. */
const withEnqueueNotification = (
  state: WorkbenchState,
  nextState: WorkbenchState,
  projectId: string | null
): WorkbenchState => {
  const before = state.projects.find((project) => project.id === projectId);
  const after = nextState.projects.find((project) => project.id === projectId);

  if (!before || !after || before.queue.items.length >= after.queue.items.length) {
    return nextState;
  }

  const queueItem = after.queue.items[0];

  const metadataOmitted =
    queueItem?.snapshot.sourceId === 'workflow' &&
    queueItem.snapshot.backendSubmission.kind === 'workflow' &&
    !queueItem.snapshot.backendSubmission.workflow;

  const withMetadataNotice = metadataOmitted
    ? addNotification(
        nextState,
        createNotification({
          kind: 'info',
          message: 'Workflow metadata was omitted because the workflow contains multiple workflow_return nodes.',
          messageKey: 'workflowLibrary.workflowMetadataOmittedBody',
          projectId: after.id,
          title: 'Workflow metadata omitted',
          titleKey: 'workflowLibrary.workflowMetadataOmitted',
        })
      )
    : nextState;

  return addNotification(
    withMetadataNotice,
    createNotification({
      category: 'enqueue',
      kind: 'success',
      message: `${after.name}: ${queueItem.snapshot.sourceId} to ${queueItem.snapshot.destination}`,
      projectId: after.id,
      title: 'Invocation queued',
    })
  );
};

const areRecordsShallowEqual = (left: Record<string, unknown>, right: Record<string, unknown>): boolean => {
  if (left === right) {
    return true;
  }

  const leftKeys = Object.keys(left);
  const rightKeys = Object.keys(right);

  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every((key) => Object.prototype.hasOwnProperty.call(right, key) && Object.is(left[key], right[key]))
  );
};

const areProjectSettingValuesEqual = (
  left: ProjectSettings[keyof ProjectSettings],
  right: ProjectSettings[keyof ProjectSettings]
): boolean => {
  if (Array.isArray(left) && Array.isArray(right)) {
    return left.length === right.length && left.every((value, index) => value === right[index]);
  }

  return Object.is(left, right);
};

const patchRecord = <RecordValue extends Record<string, unknown>>(
  current: RecordValue,
  patch: Partial<RecordValue>
): RecordValue => {
  let didChange = false;

  for (const [key, value] of Object.entries(patch)) {
    if (!Object.prototype.hasOwnProperty.call(current, key) || !Object.is(current[key], value)) {
      didChange = true;
      break;
    }
  }

  return didChange ? ({ ...current, ...patch } as RecordValue) : current;
};

const cloneRecord = <RecordValue extends Record<string, unknown>>(record: RecordValue): RecordValue =>
  structuredClone(record) as RecordValue;

const cloneGraph = (graph: GraphContract): GraphContract => ({
  ...graph,
  backendGraph: graph.backendGraph
    ? {
        ...graph.backendGraph,
        edges: graph.backendGraph.edges.map((edge) => ({
          destination: { ...edge.destination },
          source: { ...edge.source },
          ...(edge.type ? { type: edge.type } : {}),
        })),
        nodes: Object.fromEntries(Object.entries(graph.backendGraph.nodes).map(([id, node]) => [id, { ...node }])),
      }
    : undefined,
  edges: graph.edges.map((edge) => ({ ...edge })),
  nodes: graph.nodes.map((node) => ({ ...node, inputs: { ...node.inputs } })),
});

const applyQueueGenerateSnapshotToWidgetStates = (
  widgetStates: WidgetStateMap,
  generate: QueueGenerateSnapshot | undefined
): WidgetStateMap => {
  if (!generate) {
    return widgetStates;
  }

  const generateState = widgetStates.generate ?? { id: 'generate', label: 'Generate', values: {}, version: 1 as const };

  return {
    ...widgetStates,
    generate: {
      ...generateState,
      values: cloneGenerateWidgetValues(generate.values),
    },
  };
};

const clonePlacement = (placement: CanvasPlacementContract): CanvasPlacementContract => ({ ...placement });

const createCenteredPlacement = (
  image: Pick<GeneratedImageContract, 'height' | 'width'>,
  document: Pick<CanvasDocumentContractV3, 'height' | 'width'>
): CanvasPlacementContract => {
  const imageWidth = image.width > 0 ? image.width : document.width;
  const imageHeight = image.height > 0 ? image.height : document.height;
  const scale = Math.min(document.width / imageWidth, document.height / imageHeight);
  const width = Math.round(imageWidth * scale);
  const height = Math.round(imageHeight * scale);

  return {
    height,
    opacity: 1,
    width,
    x: Math.round((document.width - width) / 2),
    y: Math.round((document.height - height) / 2),
  };
};

const normalizeStagingCandidate = (
  image: CanvasStagingCandidateContract | GeneratedImageContract,
  document: Pick<CanvasDocumentContractV3, 'height' | 'width'>,
  sourceBackendItemId?: number
): CanvasStagingCandidateContract => ({
  ...image,
  ...('sourceBackendItemId' in image && image.sourceBackendItemId !== undefined
    ? { sourceBackendItemId: image.sourceBackendItemId }
    : sourceBackendItemId === undefined
      ? {}
      : { sourceBackendItemId }),
  placement:
    'placement' in image && image.placement
      ? clonePlacement(image.placement)
      : createCenteredPlacement(image, document),
});

const clampStagedImageIndex = (imageIndex: number, pendingImageCount: number): number => {
  const maxIndex = Math.max(0, pendingImageCount - 1);

  return Math.min(maxIndex, Math.max(0, imageIndex));
};

const getCanvasStagingSlotCountWithPendingImages = (
  project: Project,
  pendingImages: CanvasStagingCandidateContract[]
): number =>
  getCanvasStagingSlotCount(
    {
      ...project.canvas,
      stagingArea: {
        ...project.canvas.stagingArea,
        pendingImages,
      },
    },
    project.queue.items
  );

const getCanvasWithPendingImages = (
  canvas: CanvasStateContractV3,
  pendingImages: CanvasStagingCandidateContract[]
): CanvasStateContractV3 => ({
  ...canvas,
  stagingArea: {
    ...canvas.stagingArea,
    pendingImages,
  },
});

const getSelectedCanvasStagingSlot = (project: Project): CanvasStagingSlot | undefined =>
  getCanvasStagingSlots(project.canvas, project.queue.items)[project.canvas.stagingArea.selectedImageIndex];

const findPreservedCanvasStagingSlotIndex = (
  slots: CanvasStagingSlot[],
  selectedSlot: CanvasStagingSlot | undefined
): number => {
  if (!selectedSlot) {
    return -1;
  }

  if (selectedSlot.kind === 'candidate') {
    const exactCandidateIndex = slots.findIndex(
      (slot) => slot.kind === 'candidate' && slot.candidate === selectedSlot.candidate
    );

    if (exactCandidateIndex !== -1) {
      return exactCandidateIndex;
    }
  }

  const exactIdIndex = slots.findIndex((slot) => slot.id === selectedSlot.id);

  if (exactIdIndex !== -1) {
    return exactIdIndex;
  }

  if (selectedSlot.itemIndex === undefined) {
    return -1;
  }

  return slots.findIndex(
    (slot) => slot.queueItemId === selectedSlot.queueItemId && slot.itemIndex === selectedSlot.itemIndex
  );
};

const getCanvasStagingCandidateSlotIndex = (
  project: Project,
  pendingImages: CanvasStagingCandidateContract[],
  target: CanvasStagingCandidateContract | undefined
): number => {
  if (!target) {
    return -1;
  }

  const slots = getCanvasStagingSlots(getCanvasWithPendingImages(project.canvas, pendingImages), project.queue.items);
  const exactIndex = slots.findIndex((slot) => slot.kind === 'candidate' && slot.candidate === target);

  if (exactIndex !== -1) {
    return exactIndex;
  }

  return slots.findIndex(
    (slot) =>
      slot.kind === 'candidate' &&
      slot.candidate.sourceQueueItemId === target.sourceQueueItemId &&
      slot.candidate.imageName === target.imageName
  );
};

const resolveStagingSelectionIndexForSlots = ({
  incomingImages,
  pendingImages,
  previousSelectedSlot,
  project,
  slotCount,
}: {
  incomingImages: CanvasStagingCandidateContract[];
  pendingImages: CanvasStagingCandidateContract[];
  previousSelectedSlot?: CanvasStagingSlot;
  project: Project;
  slotCount: number;
}): number => {
  if (project.canvas.stagingArea.autoSwitchMode === 'progress') {
    const placeholderIndex = getFirstCanvasPlaceholderSlotIndex(
      getCanvasWithPendingImages(project.canvas, pendingImages),
      project.queue.items
    );

    if (placeholderIndex !== -1) {
      return placeholderIndex;
    }

    const firstIncomingSlotIndex = getCanvasStagingCandidateSlotIndex(project, pendingImages, incomingImages[0]);

    return firstIncomingSlotIndex !== -1
      ? firstIncomingSlotIndex
      : clampStagedImageIndex(project.canvas.stagingArea.selectedImageIndex, slotCount);
  }

  if (project.canvas.stagingArea.autoSwitchMode === 'off' || incomingImages.length === 0) {
    const selectedSlot = previousSelectedSlot ?? getSelectedCanvasStagingSlot(project);
    const preservedSlotIndex = findPreservedCanvasStagingSlotIndex(
      getCanvasStagingSlots(getCanvasWithPendingImages(project.canvas, pendingImages), project.queue.items),
      selectedSlot
    );

    if (preservedSlotIndex !== -1) {
      return preservedSlotIndex;
    }

    return clampStagedImageIndex(project.canvas.stagingArea.selectedImageIndex, slotCount);
  }

  const selectedImage = pendingImages[pendingImages.length - 1] ?? incomingImages[incomingImages.length - 1];
  const selectedSlotIndex = getCanvasStagingCandidateSlotIndex(project, pendingImages, selectedImage);

  return selectedSlotIndex === -1
    ? clampStagedImageIndex(project.canvas.stagingArea.selectedImageIndex, slotCount)
    : selectedSlotIndex;
};

const stageCanvasResultImages = (
  project: Project,
  queueItemId: string,
  images: GeneratedImageContract[],
  sourceBackendItemIds?: readonly (number | undefined)[],
  previousSelectedSlot?: CanvasStagingSlot
): Project => {
  const queueItem = project.queue.items.find((item) => item.id === queueItemId);

  if (
    images.length === 0 ||
    !queueItem ||
    queueItem.snapshot.canvas.documentRevision !== project.canvas.documentRevision
  ) {
    return project;
  }

  const queueDocument = queueItem.snapshot.canvas.document;
  const { bbox } = queueDocument;
  const incomingImages = images.map((image, index) =>
    normalizeStagingCandidate(
      {
        ...image,
        placement: { height: image.height, opacity: 1, width: image.width, x: bbox.x, y: bbox.y },
      },
      queueDocument,
      sourceBackendItemIds?.[index]
    )
  );
  const existingImages = project.canvas.stagingArea.pendingImages;
  const existingImageKeys = new Set(existingImages.map((image) => `${image.sourceQueueItemId}:${image.imageName}`));
  const newImages = incomingImages.filter(
    (image) => !existingImageKeys.has(`${image.sourceQueueItemId}:${image.imageName}`)
  );
  const pendingImages = [...existingImages, ...newImages];
  const slotCount = getCanvasStagingSlotCountWithPendingImages(project, pendingImages);

  return {
    ...project,
    canvas: {
      ...project.canvas,
      stagingArea: {
        ...project.canvas.stagingArea,
        areThumbnailsVisible: true,
        isVisible: slotCount > 0,
        pendingImageIds: pendingImages.map((image) => image.imageName),
        pendingImages,
        selectedImageIndex: resolveStagingSelectionIndexForSlots({
          incomingImages: newImages,
          pendingImages,
          previousSelectedSlot,
          project,
          slotCount,
        }),
        sourceQueueItemId: queueItemId,
      },
    },
  };
};

const appendCanvasStagingCandidate = (project: Project, candidate: CanvasStagingCandidateContract): Project => {
  const appendedCandidate = normalizeStagingCandidate(candidate, project.canvas.document);
  const pendingImages = [...project.canvas.stagingArea.pendingImages, appendedCandidate];
  const slots = getCanvasStagingSlots(getCanvasWithPendingImages(project.canvas, pendingImages), project.queue.items);
  const selectedImageIndex = resolveStagingSelectionIndexForSlots({
    incomingImages: [appendedCandidate],
    pendingImages,
    project,
    slotCount: slots.length,
  });

  return {
    ...project,
    canvas: {
      ...project.canvas,
      stagingArea: {
        ...project.canvas.stagingArea,
        areThumbnailsVisible: true,
        isVisible: true,
        pendingImageIds: pendingImages.map((image) => image.imageName),
        pendingImages,
        selectedImageIndex,
        sourceQueueItemId: appendedCandidate.sourceQueueItemId,
      },
    },
  };
};

const clampCanvasStagingSelection = (project: Project, previousSelectedSlot?: CanvasStagingSlot): Project => {
  const slots = getCanvasStagingSlots(project.canvas, project.queue.items);
  const slotCount = slots.length;
  const placeholderIndex =
    project.canvas.stagingArea.autoSwitchMode === 'progress'
      ? getFirstCanvasPlaceholderSlotIndex(project.canvas, project.queue.items)
      : -1;
  const preservedSlotIndex = findPreservedCanvasStagingSlotIndex(slots, previousSelectedSlot);
  const selectedImageIndex =
    placeholderIndex !== -1
      ? placeholderIndex
      : preservedSlotIndex !== -1
        ? preservedSlotIndex
        : clampStagedImageIndex(project.canvas.stagingArea.selectedImageIndex, slotCount);
  const isVisible = slotCount > 0 ? project.canvas.stagingArea.isVisible : false;

  if (
    selectedImageIndex === project.canvas.stagingArea.selectedImageIndex &&
    isVisible === project.canvas.stagingArea.isVisible
  ) {
    return project;
  }

  return {
    ...project,
    canvas: {
      ...project.canvas,
      stagingArea: {
        ...project.canvas.stagingArea,
        isVisible,
        selectedImageIndex,
      },
    },
  };
};

const getGalleryImages = (values: Record<string, unknown>): GeneratedImageContract[] =>
  getBoundedRecentImages(values.recentImages);

const canonicalizeGalleryItemKey = (key: string): GalleryItemKey => toGalleryItemKey(parseGalleryItemKey(key));

const getGalleryItemFromPersistedValue = (values: Record<string, unknown>, value: unknown): GalleryItem | null =>
  getSelectedGalleryItemFromValues({
    selectedBoardId: values.selectedBoardId,
    selectedImage: value,
    selectedImageName: null,
  });

/**
 * Deep-clones an already-v2 canvas state and normalizes staging candidate placements. Not a
 * migration boundary: callers with genuinely unknown/legacy input must run
 * `loadCanvasState` first (see `normalizeWorkbenchProject`).
 */
const cloneCanvas = (canvas: CanvasStateContractV3): CanvasStateContractV3 => {
  const document = structuredClone(canvas.document);

  return {
    version: canvas.version,
    document,
    documentRevision: canvas.documentRevision,
    snapshots: canvas.snapshots.map((snapshot) => ({ ...snapshot, document: structuredClone(snapshot.document) })),
    stagingArea: {
      ...canvas.stagingArea,
      pendingImageIds: [...(canvas.stagingArea?.pendingImageIds ?? [])],
      pendingImages: (canvas.stagingArea?.pendingImages ?? []).map((image) =>
        normalizeStagingCandidate(image, document)
      ),
      areThumbnailsVisible: canvas.stagingArea?.areThumbnailsVisible ?? true,
      autoSwitchMode: canvas.stagingArea?.autoSwitchMode ?? 'off',
      isVisible: canvas.stagingArea?.isVisible ?? (canvas.stagingArea?.pendingImages?.length ?? 0) > 0,
      selectedImageIndex: canvas.stagingArea?.selectedImageIndex ?? 0,
    },
  };
};

const cloneWidgetState = (widgetState: WidgetStateContract): WidgetStateContract => ({
  ...widgetState,
  values: { ...widgetState.values },
});

const cloneWidgetInstance = (widgetInstance: WidgetInstanceContract): WidgetInstanceContract => ({
  ...widgetInstance,
  state: cloneWidgetState(widgetInstance.state),
});

const cloneWidgetInstances = (
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>
): Record<WidgetInstanceId, WidgetInstanceContract> =>
  Object.fromEntries(
    Object.entries({ ...createWidgetInstances(), ...widgetInstances }).map(([instanceId, widgetInstance]) => [
      instanceId,
      cloneWidgetInstance(widgetInstance),
    ])
  );

const getWidgetStatesSnapshot = (widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>): WidgetStateMap => {
  const widgetStates: WidgetStateMap = {};

  for (const widgetInstance of Object.values(widgetInstances)) {
    widgetStates[widgetInstance.typeId] ??= cloneWidgetState(widgetInstance.state);
  }

  return widgetStates;
};

const getWidgetState = (project: Project, widgetId: WidgetTypeId): WidgetStateContract => {
  const widgetInstance =
    project.widgetInstances[widgetId] ??
    Object.values(project.widgetInstances).find((instance) => instance.typeId === widgetId);

  return widgetInstance?.state ?? createWidgetState(widgetId);
};

const getWidgetValues = (project: Project, widgetId: WidgetTypeId): Record<string, unknown> =>
  getWidgetState(project, widgetId).values;

const updateProjectWidgetState = (
  project: Project,
  widgetId: WidgetTypeId,
  getState: (state: WidgetStateContract) => WidgetStateContract
): Project => {
  const instance =
    project.widgetInstances[widgetId] ??
    Object.values(project.widgetInstances).find((candidate) => candidate.typeId === widgetId);
  const instanceId = instance?.id ?? widgetId;
  const currentInstance = instance ?? createWidgetInstance(widgetId, instanceId);
  const nextState = getState(currentInstance.state);

  if (nextState === currentInstance.state) {
    return project;
  }

  return {
    ...project,
    widgetInstances: {
      ...project.widgetInstances,
      [instanceId]: {
        ...currentInstance,
        state: nextState,
      },
    },
  };
};

const updateProjectWidgetValues = (
  project: Project,
  widgetId: WidgetTypeId,
  getValues: (values: Record<string, unknown>) => Record<string, unknown>
): Project =>
  updateProjectWidgetState(project, widgetId, (widgetState) => {
    const values = getValues(widgetState.values);

    return values === widgetState.values ? widgetState : { ...widgetState, values };
  });

const updateProjectWidgetInstanceValues = (
  project: Project,
  instanceId: WidgetInstanceId,
  getValues: (values: Record<string, unknown>) => Record<string, unknown>
): Project => {
  const instance = project.widgetInstances[instanceId];

  if (!instance) {
    return project;
  }

  const values = getValues(instance.state.values);

  if (values === instance.state.values) {
    return project;
  }

  return {
    ...project,
    widgetInstances: {
      ...project.widgetInstances,
      [instanceId]: {
        ...instance,
        state: { ...instance.state, values },
      },
    },
  };
};

const cloneWidgetRegions = cloneLayoutPresetWidgetRegions;

const cloneWidgetGraphs = (widgetGraphs: Project['widgetGraphs']): Project['widgetGraphs'] =>
  Object.fromEntries(Object.entries(widgetGraphs).map(([key, graph]) => [key, graph ? cloneGraph(graph) : graph]));

// Canvas pixel history belongs to the engine and graph history to each workflow; project undo preserves both.
const createUndoSnapshot = (project: Project): ProjectUndoSnapshot => ({
  floatingWidgets: project.floatingWidgets ? { ...project.floatingWidgets } : undefined,
  invocation: { ...project.invocation },
  layout: { ...project.layout, panels: { ...project.layout.panels } },
  // Working copies are never mutated in place, so the snapshot shares them rather than copying.
  presetWorkingLayouts: project.presetWorkingLayouts,
  widgetGraphs: cloneWidgetGraphs(project.widgetGraphs),
  widgetInstances: cloneWidgetInstances(project.widgetInstances),
  widgetRegions: cloneWidgetRegions(project.widgetRegions),
});

const restoreUndoSnapshot = (project: Project, snapshot: ProjectUndoSnapshot): Project => ({
  ...project,
  // Restore floating windows with widgetRegions to avoid duplicate or orphaned placements.
  floatingWidgets: snapshot.floatingWidgets ? { ...snapshot.floatingWidgets } : undefined,
  // The memory is not part of the snapshot, but a window the undo closes is still remembered.
  lastFloatingGeometry: rememberClosedWindows(
    project.lastFloatingGeometry,
    project.floatingWidgets,
    snapshot.floatingWidgets,
    // The snapshot's instances replace the project's, so an instance created since is gone.
    (instanceId) => Object.hasOwn(snapshot.widgetInstances, instanceId)
  ),
  invocation: { ...snapshot.invocation },
  layout: { ...snapshot.layout, panels: { ...snapshot.layout.panels } },
  presetWorkingLayouts: snapshot.presetWorkingLayouts,
  widgetGraphs: cloneWidgetGraphs(snapshot.widgetGraphs),
  widgetInstances: cloneWidgetInstances(snapshot.widgetInstances),
  widgetRegions: cloneWidgetRegions(snapshot.widgetRegions),
});

/** Capture pre-edit state; edits sharing a mergeKey within the window undo as one burst. */
const pushUndo = (project: Project, label: string, mergeKey?: string): Project => {
  const previous = project.undoRedo.past.at(-1);
  const timestamp = now();

  // An undo in between (`future` non-empty) ends the burst: the state the user
  // just stood on must stay reachable as its own step.
  if (
    mergeKey &&
    previous?.mergeKey === mergeKey &&
    project.undoRedo.future.length === 0 &&
    Date.parse(timestamp) - Date.parse(previous.mergedAt ?? previous.createdAt) <= UNDO_MERGE_WINDOW_MS
  ) {
    return {
      ...project,
      undoRedo: {
        future: [],
        past: [...project.undoRedo.past.slice(0, -1), { ...previous, mergedAt: timestamp }],
      },
    };
  }

  return {
    ...project,
    undoRedo: {
      future: [],
      past: [
        ...project.undoRedo.past,
        {
          createdAt: timestamp,
          id: createId('undo'),
          label,
          ...(mergeKey ? { mergeKey } : {}),
          project: createUndoSnapshot(project),
        },
      ].slice(-HISTORY_LIMIT),
    },
  };
};

const createWidgetStates = (): WidgetStateMap => ({
  'autosave-status': { id: 'autosave-status', label: 'Autosave', values: {}, version: 1 },
  canvas: { id: 'canvas', label: 'Canvas', values: {}, version: 1 },
  diagnostics: { id: 'diagnostics', label: 'Diagnostics', values: {}, version: 1 },
  gallery: { id: 'gallery', label: 'Gallery', values: {}, version: 1 },
  generate: { graphId: 'generate-graph', id: 'generate', label: 'Generate', values: {}, version: 1 },
  'image-map': { id: 'image-map', label: 'Image Map', values: {}, version: 1 },
  layers: { id: 'layers', label: 'Layers', values: {}, version: 1 },
  notifications: { id: 'notifications', label: 'Notifications', values: {}, version: 1 },
  preview: { id: 'preview', label: 'Preview', values: {}, version: 1 },
  project: { id: 'project', label: 'Project', values: {}, version: 1 },
  queue: { id: 'queue', label: 'Queue', values: {}, version: 1 },
  'server-status': { id: 'server-status', label: 'Server Status', values: {}, version: 1 },
  users: { id: 'users', label: 'Users', values: {}, version: 1 },
  workflow: { graphId: 'workflow-graph', id: 'workflow', label: 'Workflow', values: { batchCount: 1 }, version: 1 },
  upscale: { graphId: 'upscale-graph', id: 'upscale', label: 'Upscale', values: {}, version: 1 },
  video: { graphId: 'video-graph', id: 'video', label: 'Video', values: {}, version: 1 },
});

const createWidgetState = (widgetId: WidgetTypeId): WidgetStateContract =>
  cloneWidgetState(
    createWidgetStates()[widgetId] ?? {
      id: widgetId,
      label: widgetId,
      values: {},
      version: 1,
    }
  );

const createWidgetInstance = (
  widgetId: WidgetTypeId,
  instanceId: WidgetInstanceId = widgetId,
  values?: Record<string, unknown>
): WidgetInstanceContract => ({
  createdAt: now(),
  id: instanceId,
  state: values ? { ...createWidgetState(widgetId), values } : createWidgetState(widgetId),
  typeId: widgetId,
});

const defaultWidgetInstanceTypes: Record<WidgetInstanceId, WidgetTypeId> = {
  shortcuts: 'shortcuts',
  'autosave-status': 'autosave-status',
  canvas: 'canvas',
  diagnostics: 'diagnostics',
  'diagnostics:bottom': 'diagnostics',
  gallery: 'gallery',
  'gallery:bottom': 'gallery',
  'gallery:center': 'gallery',
  generate: 'generate',
  'image-map': 'image-map',
  upscale: 'upscale',
  video: 'video',
  layers: 'layers',
  notifications: 'notifications',
  preview: 'preview',
  project: 'project',
  queue: 'queue',
  'server-status': 'server-status',
  workflow: 'workflow',
  'workflow:bottom': 'workflow',
  'workflow:center': 'workflow',
};

const createWidgetInstances = (): Record<WidgetInstanceId, WidgetInstanceContract> =>
  Object.fromEntries(
    Object.entries(defaultWidgetInstanceTypes).map(([instanceId, widgetId]) => [
      instanceId,
      createWidgetInstance(widgetId, instanceId),
    ])
  );

const createWidgetRegions = (): Record<WidgetRegion, WidgetRegionState> => ({
  ...cloneLayoutPresetWidgetRegions(defaultLayoutPreset.snapshot.widgetRegions),
});

const LEGACY_DEFAULT_LEFT_REGION_WIDGET_IDS: readonly WidgetInstanceId[][] = [
  ['generate', 'workflow'],
  ['workflow', 'generate'],
  ['generate', 'workflow', 'gallery'],
];

const ensureLeftRegion = (leftRegion: WidgetRegionState | undefined): WidgetRegionState => {
  const fallback = createWidgetRegions().left;

  if (!leftRegion) {
    return fallback;
  }
  let region = leftRegion;

  if (!region.instanceIds.includes('upscale')) {
    const legacyMatch = LEGACY_DEFAULT_LEFT_REGION_WIDGET_IDS.some(
      (ids) => ids.length === region.instanceIds.length && ids.every((id, index) => region.instanceIds[index] === id)
    );

    if (legacyMatch) {
      const galleryIndex = region.instanceIds.indexOf('gallery');
      const instanceIds = [...region.instanceIds];

      instanceIds.splice(galleryIndex === -1 ? instanceIds.length : galleryIndex, 0, 'upscale');
      region = { ...region, instanceIds };
    }
  }

  return region;
};

// Exact historical defaults adopt the current rail wholesale; splicing would incorrectly mark untouched layouts as
// customized.
const LEGACY_RIGHT_REGION_WIDGET_IDS: WidgetId[][] = [
  ['queue', 'gallery', 'layers'],
  // The rail as it shipped before the image map existed.
  ['gallery', 'preview', 'queue', 'layers', 'diagnostics', 'project'],
];

const isLegacyDefaultRightRegion = (region: WidgetRegionState): boolean =>
  LEGACY_RIGHT_REGION_WIDGET_IDS.some(
    (ids) =>
      ids.length === region.instanceIds.length && ids.every((widgetId, index) => region.instanceIds[index] === widgetId)
  );

const ensureRightRegion = (rightRegion: WidgetRegionState | undefined): WidgetRegionState => {
  const defaultRightRegion = createWidgetRegions().right;

  if (!rightRegion) {
    return defaultRightRegion;
  }

  if (isLegacyDefaultRightRegion(rightRegion)) {
    return { ...rightRegion, instanceIds: defaultRightRegion.instanceIds };
  }

  return rightRegion;
};

/** Exact historical Edit defaults adopt the current rail; customized rails remain unchanged. */
const LEGACY_EDIT_RIGHT_REGION_WIDGET_IDS: ReadonlyArray<readonly WidgetInstanceId[]> = [
  ['layers', 'preview', 'gallery', 'image-map', 'queue'],
  ['layers', 'preview', 'gallery', 'queue'],
  ['layers'],
];

const sameInstanceIds = (region: WidgetRegionState, ids: readonly WidgetInstanceId[]): boolean =>
  region.instanceIds.length === ids.length && region.instanceIds.every((id, index) => id === ids[index]);

const ensureEditRightRegion = (right: WidgetRegionState): WidgetRegionState => {
  if (!LEGACY_EDIT_RIGHT_REGION_WIDGET_IDS.some((ids) => sameInstanceIds(right, ids))) {
    return right;
  }
  const edit = getLayoutPreset('edit').snapshot.widgetRegions.right;
  return { ...right, activeInstanceId: edit.activeInstanceId, instanceIds: [...edit.instanceIds] };
};

/** The canvas editors that folded into the Layers panel; anything persisted about them drops on load. */
const RETIRED_WIDGET_TYPE_IDS: ReadonlySet<string> = new Set(['properties', 'transform']);

const withoutRetiredInstances = (
  region: WidgetRegionState,
  retired: ReadonlySet<WidgetInstanceId>
): WidgetRegionState => {
  if (!region.instanceIds.some((instanceId) => retired.has(instanceId))) {
    return region;
  }
  const instanceIds = region.instanceIds.filter((instanceId) => !retired.has(instanceId));
  return {
    ...region,
    activeInstanceId: retired.has(region.activeInstanceId) ? (instanceIds[0] ?? '') : region.activeInstanceId,
    instanceIds,
  };
};

// Adopt queue-status for the historical bottom default. Placement normalization removes any instance already
// hosted in a window.
const LEGACY_DEFAULT_BOTTOM_REGION_WIDGET_IDS: readonly WidgetInstanceId[] = [
  'server-status',
  'gallery:bottom',
  'notifications',
  'autosave-status',
];

const isLegacyDefaultBottomRegion = (region: WidgetRegionState): boolean =>
  region.instanceIds.length === LEGACY_DEFAULT_BOTTOM_REGION_WIDGET_IDS.length &&
  region.instanceIds.every((widgetId, index) => widgetId === LEGACY_DEFAULT_BOTTOM_REGION_WIDGET_IDS[index]);

const ensureBottomRegion = (bottomRegion: WidgetRegionState | undefined): WidgetRegionState => {
  const fallback = createWidgetRegions().bottom;

  if (!bottomRegion) {
    return fallback;
  }
  if (bottomRegion.instanceIds.includes('queue-status')) {
    return bottomRegion;
  }

  if (!isLegacyDefaultBottomRegion(bottomRegion)) {
    return bottomRegion;
  }

  const serverStatusIndex = bottomRegion.instanceIds.indexOf('server-status');
  const instanceIds = [...bottomRegion.instanceIds];

  instanceIds.splice(serverStatusIndex === -1 ? instanceIds.length : serverStatusIndex + 1, 0, 'queue-status');

  return { ...bottomRegion, instanceIds };
};

const getCenterWidgetIdFromViewId = (centerViewId: CenterViewId): WidgetInstanceId => {
  if (centerViewId === 'gallery') {
    return 'gallery:center';
  }

  if (centerViewId === 'workflow') {
    return 'workflow:center';
  }

  return centerViewId;
};

const ensureCenterRegion = (
  centerRegion: WidgetRegionState | undefined,
  fallbackCenterViewId: CenterViewId
): WidgetRegionState => {
  const defaultCenterRegion = createWidgetRegions().center;
  // Missing region data adopts defaults; an explicitly empty center may have its last view floating and must stay
  // empty.
  const instanceIds = centerRegion ? centerRegion.instanceIds : defaultCenterRegion.instanceIds;
  const activeInstanceId = centerRegion?.activeInstanceId ?? getCenterWidgetIdFromViewId(fallbackCenterViewId);
  // An empty center retains the floated instance pointer for boot preloading; nonempty centers clamp invalid
  // pointers.
  const normalizedActiveInstanceId = instanceIds.includes(activeInstanceId)
    ? activeInstanceId
    : (instanceIds[0] ?? activeInstanceId);

  return {
    ...defaultCenterRegion,
    ...centerRegion,
    activeInstanceId: normalizedActiveInstanceId,
    instanceIds,
    isCollapsed: false,
  };
};

const normalizePromptHistory = (value: unknown): PromptHistoryItem[] => {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.reduceRight<PromptHistoryItem[]>((history, item) => {
    if (!item || typeof item !== 'object') {
      return history;
    }

    const record = item as Record<string, unknown>;

    if (typeof record.positivePrompt !== 'string') {
      return history;
    }

    return addPromptHistoryItem(history, {
      negativePrompt: typeof record.negativePrompt === 'string' ? record.negativePrompt : null,
      positivePrompt: record.positivePrompt,
    });
  }, []);
};

/** The project-ingestion path: gate every embedded canvas, then normalize. */
export const loadWorkbenchProject = (raw: Project): ProjectLoadResult => {
  const refused = gateProjectCanvases(raw);

  return refused ? { refused, status: 'refused' } : { project: normalizeWorkbenchProject(raw), status: 'loaded' };
};

/** Normalizes an admitted project; a canvas that somehow fails to reload is kept as-is, never rewritten. */
export const normalizeWorkbenchProject = (
  project: Project,
  options: {
    /**
     * Drop session-only infinite-window anchors on imported/server documents; preserve them during live
     * retargeting.
     */
    isArriving?: boolean;
  } = {}
): Project => {
  const canvas = loadCanvasState(project.canvas);

  return assembleWorkbenchProject(
    project,
    cloneCanvas(canvas.status === 'loaded' ? canvas.value : project.canvas),
    options
  );
};

const assembleWorkbenchProject = (
  project: Project,
  canvas: CanvasStateContractV3,
  options: {
    isArriving?: boolean;
  } = {}
): Project => {
  const { isArriving = true } = options;
  const {
    graphHistory: _graphHistory,
    projectGraph: legacyProjectGraph,
    recoveredAt: _recoveredAt,
    recoveryOf: _recoveryOf,
    ...persistentProject
  } = project as Project & {
    graphHistory?: unknown;
    projectGraph?: unknown;
    recoveredAt?: unknown;
    recoveryOf?: unknown;
  };
  const legacyWidgetRegions = project.widgetRegions as
    | Partial<Record<WidgetRegion | 'left-panel' | 'right-panel' | 'status-bar', WidgetRegionState>>
    | undefined;
  const leftRegion = ensureLeftRegion(legacyWidgetRegions?.left ?? legacyWidgetRegions?.['left-panel']);
  const bottomRegion = ensureBottomRegion(legacyWidgetRegions?.bottom ?? legacyWidgetRegions?.['status-bar']);
  const widgetInstances = cloneWidgetInstances(project.widgetInstances ?? createWidgetInstances());

  const generateInstance = widgetInstances.generate;
  const upscaleInstance = widgetInstances.upscale;

  if (generateInstance && upscaleInstance) {
    const migratedValues = migrateProjectPromptDraft(generateInstance.state.values, upscaleInstance.state.values);
    const clearedLegacyUpscaleValues = applyProjectPromptDraft(upscaleInstance.state.values, {
      negativePrompt: '',
      negativePromptEnabled: true,
      positivePrompt: '',
    });

    if (migratedValues !== generateInstance.state.values) {
      widgetInstances.generate = {
        ...generateInstance,
        state: { ...generateInstance.state, values: migratedValues },
      };
    }

    if (clearedLegacyUpscaleValues !== upscaleInstance.state.values) {
      widgetInstances.upscale = {
        ...upscaleInstance,
        state: { ...upscaleInstance.state, values: clearedLegacyUpscaleValues },
      };
    }
  }

  // Legacy workflows inherit Generate's iteration count once; fresh workflows use their own default.
  const workflowInstance = widgetInstances.workflow;

  if (workflowInstance && typeof workflowInstance.state.values.batchCount !== 'number') {
    widgetInstances.workflow = {
      ...workflowInstance,
      state: {
        ...workflowInstance.state,
        values: {
          ...workflowInstance.state.values,
          batchCount: sanitizeBatchCount(generateInstance?.state.values.batchCount),
        },
      },
    };
  }

  if (leftRegion.instanceIds.includes('upscale') && !widgetInstances.upscale) {
    widgetInstances.upscale = createWidgetInstance('upscale');
  }
  if (leftRegion.instanceIds.includes('video') && !widgetInstances.video) {
    widgetInstances.video = createWidgetInstance('video');
  }

  if (bottomRegion.instanceIds.includes('queue-status') && !widgetInstances['queue-status']) {
    widgetInstances['queue-status'] = createWidgetInstance('queue-status');
  }

  for (const [instanceId, instance] of Object.entries(widgetInstances)) {
    if (instance.typeId !== 'gallery') {
      continue;
    }

    // Preserve only rankings resolvable in this session. Drop foreign infinite-window anchors consistently with
    // serialization so hydration matches the sync baseline.
    const strippedSearchValues = stripUnresolvableGallerySearch(instance.state.values);
    const strippedValues = isArriving
      ? (stripInfiniteWindowAnchor(strippedSearchValues ?? instance.state.values) ?? strippedSearchValues)
      : strippedSearchValues;
    const hasRecentImages = 'recentImages' in instance.state.values;

    if (strippedValues === null && !hasRecentImages) {
      continue;
    }

    const values = strippedValues ?? instance.state.values;

    widgetInstances[instanceId] = {
      ...instance,
      state: {
        ...instance.state,
        values: hasRecentImages ? { ...values, recentImages: getBoundedRecentImages(values.recentImages) } : values,
      },
    };
  }

  // The canvas editors folded into the Layers panel; an instance of the retired widgets has nothing to render.
  const retiredInstanceIds = new Set(
    Object.values(widgetInstances)
      .filter((instance) => RETIRED_WIDGET_TYPE_IDS.has(instance.typeId))
      .map((instance) => instance.id)
  );
  for (const instanceId of retiredInstanceIds) {
    delete widgetInstances[instanceId];
  }
  const rightRegion = ensureEditRightRegion(
    withoutRetiredInstances(
      ensureRightRegion(legacyWidgetRegions?.right ?? legacyWidgetRegions?.['right-panel']),
      retiredInstanceIds
    )
  );

  const placement = normalizeFloatingPlacement(
    {
      left: leftRegion,
      right: rightRegion,
      bottom: bottomRegion,
      center: ensureCenterRegion(legacyWidgetRegions?.center, project.layout.centerViewId),
    },
    (project as Partial<Project>).floatingWidgets,
    (instanceId) => Object.hasOwn(widgetInstances, instanceId)
  );

  return {
    ...persistentProject,
    canvas,
    events: isArriving ? [] : project.events.slice(0, PROJECT_EVENT_LIMIT),
    floatingWidgets: placement.floatingWidgets,
    lastFloatingGeometry: normalizeLastFloatingGeometry(
      (project as Partial<Project>).lastFloatingGeometry,
      (instanceId) => Object.hasOwn(widgetInstances, instanceId),
      placement.floatingWidgets
    ),
    // Resolve historical built-in preset ids to their current arrangements to avoid false layout drift.
    layout: { ...project.layout, presetId: resolveLayoutPresetId(project.layout.presetId) },
    presetWorkingLayouts: normalizePresetWorkingLayouts(
      (project as Partial<Project>).presetWorkingLayouts,
      resolveLayoutPresetId(project.layout.presetId)
    ),
    promptHistory: normalizePromptHistory((project as Partial<Project>).promptHistory),
    queue: isArriving ? { items: [] } : project.queue,
    settings: normalizeProjectSettings(project.settings),
    widgetRegions: placement.widgetRegions,
    widgetInstances,
    workflowHistories: isArriving ? {} : (project.workflowHistories ?? {}),
    workflows: resolveProjectWorkflows(project.workflows, legacyProjectGraph),
  };
};

/**
 * Documents reach here through `migrateProjectDocument`, which already refused malformed collections. Session
 * snapshots (`normalizeWorkbenchState`) are the one input that does not: a legacy single graph migrates, and a
 * snapshot without a usable collection gets one blank workflow rather than failing the whole session restore.
 */
const resolveProjectWorkflows = (
  candidate: ProjectWorkflowCollection | undefined,
  legacyProjectGraph: unknown
): ProjectWorkflowCollection =>
  normalizeProjectWorkflowCollection(candidate) ??
  migrateProjectGraphToCollection(legacyProjectGraph ?? {}) ??
  createProjectWorkflowCollection(createBlankWorkflowDocument());

/**
 * Assign the server board after normalization creates missing gallery instances. Projects with no gallery layout
 * remain unchanged.
 */
export const withAuthoritativeProjectBoard = (project: Project, boardId: string): Project =>
  updateProjectWidgetValues(project, 'gallery', (values) =>
    values.projectBoardId === boardId ? values : { ...values, projectBoardId: boardId }
  );

export const clampPanelSize = (region: WidgetRegion, sizePx: number): number => {
  const { max, min } = getPanelSizeBounds(region);

  return Math.min(max, Math.max(min, sizePx));
};

const createCanvasState = (): CanvasStateContractV3 => createNewCanvasState();

const createProject = (index: number, id: string, preset: LayoutPreset): Project =>
  applyLayoutPresetToProject(
    {
      canvas: createCanvasState(),
      events: [
        {
          createdAt: now(),
          id: createId('event'),
          summary: `Created Project Name #${index}`,
          type: 'project-created',
        },
      ],
      id,
      invocation: getInvocationAfterLayoutPreset(defaultInvocationRoute, preset),
      layout: { ...defaultLayoutPreset.snapshot.layout, panels: { ...defaultLayoutPreset.snapshot.layout.panels } },
      name: `Project Name #${index}`,
      promptHistory: [],
      queue: { items: [] },
      settings: normalizeProjectSettings(),
      undoRedo: { future: [], past: [] },
      widgetGraphs: {},
      widgetInstances: createWidgetInstances(),
      widgetRegions: createWidgetRegions(),
      workflowHistories: {},
      workflows: createProjectWorkflowCollection(createBlankWorkflowDocument()),
    },
    preset
  );

const getNextProjectIndex = (projects: Project[]): number => {
  const usedIndices = projects.map((project) => Number(project.name.match(/#(\d+)$/)?.[1] ?? 0));

  return Math.max(0, ...usedIndices) + 1;
};

/** Use collision-resistant ids so a draft autosave cannot overwrite an existing server project. */
export const createDraftProject = (projects: Project[], account?: WorkbenchState['account']): Project =>
  createProject(
    getNextProjectIndex(projects),
    createId('project'),
    account ? resolveSavedLayoutPreset(normalizeWorkbenchAccount(account), defaultLayoutPreset.id) : defaultLayoutPreset
  );

const updateActiveProject = (state: WorkbenchState, getProject: (project: Project) => Project): WorkbenchState => {
  let didChange = false;
  const projects = state.projects.map((project) => {
    if (project.id !== state.activeProjectId) {
      return project;
    }

    const nextProject = getProject(project);

    if (nextProject !== project) {
      didChange = true;
    }

    return nextProject;
  });

  return didChange ? { ...state, projects } : state;
};

/** Ignore empty regions so toggling panels cannot open blank panels or alter preset collapse state. */
export const resolvePanelToggle = (
  widgetRegions: Record<WidgetRegion, Pick<WidgetRegionState, 'instanceIds' | 'isCollapsed'>>,
  regions: readonly WidgetRegion[]
): { regions: WidgetRegion[]; shouldCollapse: boolean } => {
  const occupied = regions.filter((region) => widgetRegions[region].instanceIds.length > 0);
  return { regions: occupied, shouldCollapse: occupied.some((region) => !widgetRegions[region].isCollapsed) };
};

/**
 * A region with nothing left names no active instance. The center keeps naming the view it lost: that pointer is
 * what lets a floated last view come back (see `isAwaitedCenterView`).
 */
const emptiedActiveInstanceId = (regionId: WidgetRegion, region: WidgetRegionState): WidgetInstanceId =>
  regionId === 'center' ? region.activeInstanceId : '';

const getNextInstanceId = (region: WidgetRegionState, instanceId: WidgetInstanceId): WidgetInstanceId | null => {
  if (region.activeInstanceId !== instanceId) {
    return region.activeInstanceId;
  }

  return region.instanceIds.find((enabledInstanceId) => enabledInstanceId !== instanceId) ?? null;
};

/**
 * An instance opened in the center can stay a member of a side panel, but is not shown twice: a side panel fronting
 * it fronts its next member instead, or collapses when it has none. Only the pointer moves; no panel opens, so a
 * hidden panel stays hidden and later shows its neighbour. The bottom region is left alone: its members include
 * tooltip-only status items, and without the widget registry the reducer cannot tell which of them render a panel.
 */
const SIDE_PANEL_REGIONS = ['left', 'right'] as const satisfies readonly WidgetRegion[];

const yieldSidePanelsTo = (
  widgetRegions: Project['widgetRegions'],
  instanceId: WidgetInstanceId
): Project['widgetRegions'] => {
  let next = widgetRegions;

  for (const regionId of SIDE_PANEL_REGIONS) {
    const region = next[regionId];

    if (region.activeInstanceId !== instanceId) {
      continue;
    }

    const neighbour = getNextInstanceId(region, instanceId);

    next = {
      ...next,
      [regionId]: neighbour ? { ...region, activeInstanceId: neighbour } : { ...region, isCollapsed: true },
    };
  }

  return next;
};

const updateActiveWidgetRegion = (
  state: WorkbenchState,
  region: WidgetRegion,
  getRegion: (regionState: WidgetRegionState) => WidgetRegionState
): WorkbenchState => updateActiveProject(state, (project) => updateProjectWidgetRegion(project, region, getRegion));

const updateProjectWidgetRegion = (
  project: Project,
  region: WidgetRegion,
  getRegion: (regionState: WidgetRegionState) => WidgetRegionState
): Project => {
  const regionState = project.widgetRegions[region];
  const nextRegionState = getRegion(regionState);

  return nextRegionState === regionState
    ? project
    : {
        ...project,
        widgetRegions: {
          ...project.widgetRegions,
          [region]: nextRegionState,
        },
      };
};

const openPanelForRegion = (layout: ProjectLayoutState, region: WidgetRegion): ProjectLayoutState => ({
  ...layout,
  panels: {
    ...layout.panels,
    isBottomOpen: region === 'bottom' ? true : layout.panels.isBottomOpen,
    isLeftOpen: region === 'left' ? true : layout.panels.isLeftOpen,
    isRightOpen: region === 'right' ? true : layout.panels.isRightOpen,
  },
});

const withoutFloatingWidget = (
  floatingWidgets: Project['floatingWidgets'],
  instanceId: WidgetInstanceId
): Project['floatingWidgets'] => {
  const { [instanceId]: _removed, ...remaining } = floatingWidgets ?? {};

  return Object.keys(remaining).length > 0 ? remaining : undefined;
};

/**
 * Edit one region's complete order — docked members and floating markers together — so the markers' return
 * indices stay in step with the members around them. A marker the edit docks or drops takes its window with it,
 * which keeps an instance from ending up both docked and floating.
 */
const updateRegionOrder = (
  project: Project,
  regionId: WidgetRegion,
  update: (slots: RegionOrderSlot[]) => RegionOrderSlot[]
): Project => {
  const region = project.widgetRegions[regionId];
  const before = getRegionOrder(regionId, region.instanceIds, project.floatingWidgets);
  const after = update(before);
  const markerIds = new Set(after.filter((slot) => slot.isFloating).map((slot) => slot.instanceId));
  const closedIds = before
    .filter((slot) => slot.isFloating && !markerIds.has(slot.instanceId))
    .map((slot) => slot.instanceId);
  const remaining = closedIds.reduce(withoutFloatingWidget, project.floatingWidgets);
  const order = writeRegionOrder(after, remaining);
  // A window that docks or closes leaves its geometry behind, so floating the instance again reopens it there.
  const lastFloatingGeometry = rememberClosedWindows(project.lastFloatingGeometry, project.floatingWidgets, remaining);
  const isMembershipUnchanged =
    order.instanceIds.length === region.instanceIds.length &&
    order.instanceIds.every((instanceId, index) => instanceId === region.instanceIds[index]);

  // An edit that only moves markers leaves the region object alone, so its subscribers do not re-render.
  if (isMembershipUnchanged) {
    return order.floatingWidgets === project.floatingWidgets
      ? project
      : { ...project, floatingWidgets: order.floatingWidgets, lastFloatingGeometry };
  }

  return {
    ...project,
    floatingWidgets: order.floatingWidgets,
    lastFloatingGeometry,
    widgetRegions: { ...project.widgetRegions, [regionId]: { ...region, instanceIds: order.instanceIds } },
  };
};

/**
 * Put a floating instance back where it came from: its return region, at its marker's position, and the center
 * too when the center has stayed empty since the instance left it. `reveal` also fronts it in the return region.
 */
const restoreFloatingInstance = (
  project: Project,
  instanceId: WidgetInstanceId,
  { reveal }: { reveal: boolean }
): Project => {
  const floating = project.floatingWidgets?.[instanceId];

  if (!floating) {
    return project;
  }

  const { returnRegion } = floating;
  // Counting the docked members ahead of the marker keeps every other slot where the rail shows it.
  const docked = updateProjectWidgetRegion(
    updateRegionOrder(project, returnRegion, (slots) =>
      slots.map((slot) => (slot.instanceId === instanceId ? { ...slot, isFloating: false } : slot))
    ),
    returnRegion,
    (region) =>
      reveal || !region.instanceIds.includes(region.activeInstanceId)
        ? { ...region, activeInstanceId: instanceId, isCollapsed: reveal ? false : region.isCollapsed }
        : region
  );
  const restored =
    returnRegion !== 'center' && isAwaitedCenterView(project.widgetRegions.center, instanceId)
      ? updateRegionOrder(docked, 'center', (slots) => [...slots, { instanceId, isFloating: false }])
      : docked;

  return reveal ? { ...restored, layout: openPanelForRegion(restored.layout, returnRegion) } : restored;
};

/** Raise a window to the top of the stack; the same map comes back when it is missing or already topmost. */
const raiseFloatingWidget = (
  floatingWidgets: Project['floatingWidgets'],
  instanceId: WidgetInstanceId
): Project['floatingWidgets'] => {
  const floating = floatingWidgets?.[instanceId];

  if (!floatingWidgets || !floating || floating.stackOrder === nextStackOrder(floatingWidgets) - 1) {
    return floatingWidgets;
  }

  // Compact persisted stackOrder to 1..N rather than growing it on every raise.
  const below = Object.entries(floatingWidgets)
    .filter(([otherInstanceId]) => otherInstanceId !== instanceId)
    .sort(([, left], [, right]) => left.stackOrder - right.stackOrder);
  const raised: Record<WidgetInstanceId, FloatingWidgetState> = {};

  for (const [otherInstanceId, windowState] of below) {
    raised[otherInstanceId] = { ...windowState, stackOrder: Object.keys(raised).length + 1 };
  }

  raised[instanceId] = { ...floating, stackOrder: below.length + 1 };

  return raised;
};

const cloneLayoutPresetSnapshot = (snapshot: LayoutPresetSnapshot): LayoutPresetSnapshot => {
  // Strip retired editor instances when rebuilding account presets to keep applied layouts drift-free.
  const retired = new Set(
    Object.values(snapshot.widgetInstances)
      .filter((instance) => RETIRED_WIDGET_TYPE_IDS.has(instance.typeId))
      .map((instance) => instance.id)
  );
  const widgetRegions = cloneLayoutPresetWidgetRegions(snapshot.widgetRegions);
  for (const region of Object.keys(widgetRegions) as WidgetRegion[]) {
    widgetRegions[region] = withoutRetiredInstances(widgetRegions[region], retired);
  }
  // Stored placements are read the way a project's are, so an applied preset matches the snapshot it came from.
  return normalizeLayoutPresetSnapshot({
    floatingWidgets: snapshot.floatingWidgets,
    layout: { ...snapshot.layout, panels: { ...snapshot.layout.panels } },
    widgetInstances: Object.fromEntries(
      Object.entries(snapshot.widgetInstances)
        .filter(([, instance]) => !retired.has(instance.id))
        .map(([instanceId, instance]) => [instanceId, { ...instance }])
    ),
    widgetRegions,
  });
};

const centerViewIds = new Set<CenterViewId>(['canvas', 'gallery', 'preview', 'workflow']);

const isLayoutPresetWidgetInstance = (instanceId: string, value: unknown): boolean => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const record = value as { id?: unknown; title?: unknown; typeId?: unknown };

  return (
    instanceId.length > 0 &&
    record.id === instanceId &&
    typeof record.typeId === 'string' &&
    record.typeId.length > 0 &&
    (record.title === undefined || typeof record.title === 'string')
  );
};

const isWidgetRegionState = (
  value: unknown,
  widgetInstances: Readonly<Record<string, unknown>>
): value is WidgetRegionState => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const record = value as Partial<WidgetRegionState>;
  const instanceIds = record.instanceIds;

  // An empty region (a dock nothing was placed in) names no active instance.
  const isEmpty = Array.isArray(instanceIds) && instanceIds.length === 0 && record.activeInstanceId === '';

  return (
    typeof record.activeInstanceId === 'string' &&
    Array.isArray(instanceIds) &&
    instanceIds.every((instanceId) => typeof instanceId === 'string' && instanceId in widgetInstances) &&
    new Set(instanceIds).size === instanceIds.length &&
    (isEmpty ||
      (record.activeInstanceId.length > 0 &&
        record.activeInstanceId in widgetInstances &&
        (instanceIds.length === 0 || instanceIds.includes(record.activeInstanceId)))) &&
    (record.alignEndInstanceIds === undefined ||
      (Array.isArray(record.alignEndInstanceIds) &&
        record.alignEndInstanceIds.every((instanceId) => typeof instanceId === 'string'))) &&
    typeof record.isCollapsed === 'boolean' &&
    typeof record.sizePx === 'number' &&
    Number.isFinite(record.sizePx) &&
    record.sizePx >= 0
  );
};

const isLayoutPresetSnapshot = (value: unknown): value is LayoutPresetSnapshot => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const snapshot = value as Partial<LayoutPresetSnapshot>;
  const layout = snapshot.layout as Partial<ProjectLayoutState> | undefined;
  const widgetInstances = snapshot.widgetInstances as Record<string, unknown> | undefined;

  return (
    !!layout &&
    typeof layout.presetId === 'string' &&
    layout.presetId.length > 0 &&
    typeof layout.centerViewId === 'string' &&
    centerViewIds.has(layout.centerViewId as CenterViewId) &&
    !!layout.panels &&
    typeof layout.panels.isBottomOpen === 'boolean' &&
    typeof layout.panels.isLeftOpen === 'boolean' &&
    typeof layout.panels.isRightOpen === 'boolean' &&
    !!widgetInstances &&
    typeof widgetInstances === 'object' &&
    !Array.isArray(widgetInstances) &&
    Object.keys(widgetInstances).length > 0 &&
    Object.entries(widgetInstances).every(([instanceId, instance]) =>
      isLayoutPresetWidgetInstance(instanceId, instance)
    ) &&
    !!snapshot.widgetRegions &&
    typeof snapshot.widgetRegions === 'object' &&
    isWidgetRegionState(snapshot.widgetRegions.left, widgetInstances) &&
    isWidgetRegionState(snapshot.widgetRegions.right, widgetInstances) &&
    isWidgetRegionState(snapshot.widgetRegions.bottom, widgetInstances) &&
    isWidgetRegionState(snapshot.widgetRegions.center, widgetInstances)
  );
};

const normalizeLayoutPresetRoute = (value: unknown): LayoutPresetRoute | undefined => {
  if (!value || typeof value !== 'object') {
    return undefined;
  }

  const route = value as Partial<LayoutPresetRoute>;

  if (
    typeof route.sourceId !== 'string' ||
    !isInvocationSourceAvailable(route.sourceId as InvocationSourceId) ||
    typeof route.destination !== 'string' ||
    !isResultDestinationAvailable(route.destination as ResultDestination)
  ) {
    return undefined;
  }

  return { destination: route.destination as ResultDestination, sourceId: route.sourceId as InvocationSourceId };
};

const normalizeCustomLayoutPresets = (presets: unknown): LayoutPreset[] => {
  if (!Array.isArray(presets)) {
    return [];
  }

  const seenIds = new Set<string>();

  return presets.flatMap((preset): LayoutPreset[] => {
    if (!preset || typeof preset !== 'object') {
      return [];
    }

    const record = preset as Partial<LayoutPreset>;

    if (typeof record.id !== 'string' || typeof record.label !== 'string' || !isLayoutPresetSnapshot(record.snapshot)) {
      return [];
    }

    const id = record.id.trim();

    if (!id || isBuiltInLayoutPresetId(resolveLayoutPresetId(id)) || seenIds.has(id)) {
      return [];
    }

    seenIds.add(id);

    const defaultRoute = normalizeLayoutPresetRoute(record.defaultRoute);

    return [
      {
        ...(defaultRoute ? { defaultRoute } : {}),
        ...(typeof record.iconId === 'string' ? { iconId: record.iconId } : {}),
        id,
        label: record.label,
        snapshot: cloneLayoutPresetSnapshot(record.snapshot),
      },
    ];
  });
};

/**
 * How many presets' working arrangements a project keeps. Each is a few kilobytes of placement; past this the oldest
 * copy is forgotten, which loses an unsaved arrangement but never a saved preset or any widget state.
 */
export const PRESET_WORKING_LAYOUT_LIMIT = 16;

const withSnapshotPresetId = (snapshot: LayoutPresetSnapshot, presetId: LayoutPresetId): LayoutPresetSnapshot => ({
  ...snapshot,
  layout: { ...snapshot.layout, presetId },
});

const toPresetWorkingLayouts = (copies: readonly LayoutPresetWorkingCopy[]): Project['presetWorkingLayouts'] =>
  copies.length > 0 ? copies.slice(-PRESET_WORKING_LAYOUT_LIMIT) : undefined;

/** Validated as stored presets are; the active preset's copy is dropped because the live layout is that copy. */
const normalizePresetWorkingLayouts = (
  stored: unknown,
  activePresetId: LayoutPresetId
): Project['presetWorkingLayouts'] => {
  if (!Array.isArray(stored)) {
    return undefined;
  }

  const copies: LayoutPresetWorkingCopy[] = [];

  for (const entry of stored as unknown[]) {
    const { presetId: storedId, snapshot } = (entry ?? {}) as Partial<Record<keyof LayoutPresetWorkingCopy, unknown>>;
    const presetId = typeof storedId === 'string' ? resolveLayoutPresetId(storedId) : '';

    if (!presetId || presetId === activePresetId || !isLayoutPresetSnapshot(snapshot)) {
      continue;
    }

    const clone = cloneLayoutPresetSnapshot(snapshot);
    // A historical id and its current one name the same preset; the later entry is the newer copy.
    const earlier = copies.findIndex((copy) => copy.presetId === presetId);

    if (earlier !== -1) {
      copies.splice(earlier, 1);
    }
    copies.push({ presetId, snapshot: withSnapshotPresetId(clone, presetId) });
  }

  return toPresetWorkingLayouts(copies);
};

const withoutPresetWorkingLayouts = (
  project: Project,
  isDropped: (presetId: LayoutPresetId) => boolean
): Project['presetWorkingLayouts'] => {
  const copies = project.presetWorkingLayouts ?? [];
  const kept = copies.filter((copy) => !isDropped(copy.presetId));

  return kept.length === copies.length ? project.presetWorkingLayouts : toPresetWorkingLayouts(kept);
};

/**
 * Drop copies that no longer hold anything: those of presets the account no longer has (they go with the preset and
 * never take one of the bounded places), and those equal to what their preset has saved. Run where the account is
 * the one the server holds (hydration, opening a project), so a copy whose save was lost still differs and stays.
 */
const withoutRedundantWorkingCopies = (account: AccountState, project: Project): Project => {
  const presetWorkingLayouts = withoutPresetWorkingLayouts(
    project,
    (presetId) =>
      !doesLayoutPresetExist(account, presetId) ||
      areLayoutPresetSnapshotsEqual(
        findLayoutPresetWorkingCopy(project.presetWorkingLayouts, presetId)!,
        resolveSavedLayoutPreset(account, presetId).snapshot
      )
  );

  return presetWorkingLayouts === project.presetWorkingLayouts ? project : { ...project, presetWorkingLayouts };
};

/**
 * The working copies after leaving the active preset for `incomingPresetId`: the outgoing arrangement is kept,
 * newest last, when it differs from what that preset has saved; the incoming copy leaves the list because it becomes
 * the live layout; copies of presets the account no longer has are dropped.
 */
const captureOutgoingWorkingLayout = (
  account: AccountState,
  project: Project,
  incomingPresetId: LayoutPresetId
): Project['presetWorkingLayouts'] => {
  const outgoingPresetId = project.layout.presetId;
  const copies = (project.presetWorkingLayouts ?? []).filter(
    ({ presetId }) =>
      presetId !== outgoingPresetId && presetId !== incomingPresetId && doesLayoutPresetExist(account, presetId)
  );

  if (doesLayoutPresetExist(account, outgoingPresetId)) {
    const live = cloneLayoutPresetSnapshot(createLayoutPresetSnapshot(project));

    if (!areLayoutPresetSnapshotsEqual(live, resolveSavedLayoutPreset(account, outgoingPresetId).snapshot)) {
      copies.push({ presetId: outgoingPresetId, snapshot: live });
    }
  }

  return toPresetWorkingLayouts(copies);
};

const normalizeLayoutPresetRouteOverrides = (overrides: unknown): LayoutPresetRouteOverrides => {
  if (!overrides || typeof overrides !== 'object') {
    return {};
  }

  return Object.fromEntries(
    Object.entries(overrides as Record<string, unknown>).flatMap(([presetId, route]) => {
      const resolvedPresetId = resolveLayoutPresetId(presetId);
      const normalizedRoute = normalizeLayoutPresetRoute(route);

      return isBuiltInLayoutPresetId(resolvedPresetId) && normalizedRoute ? [[resolvedPresetId, normalizedRoute]] : [];
    })
  );
};

const normalizeLayoutPresetMetadataOverrides = (overrides: unknown): LayoutPresetMetadataOverrides => {
  if (!overrides || typeof overrides !== 'object') {
    return {};
  }

  return Object.fromEntries(
    Object.entries(overrides as Record<string, unknown>).flatMap(([presetId, metadata]) => {
      const resolvedPresetId = resolveLayoutPresetId(presetId);

      if (!isBuiltInLayoutPresetId(resolvedPresetId) || !metadata || typeof metadata !== 'object') {
        return [];
      }

      const record = metadata as Partial<LayoutPresetMetadataOverride>;
      const label = typeof record.label === 'string' ? record.label.trim() : '';
      const normalized: LayoutPresetMetadataOverride = {
        ...(typeof record.iconId === 'string' ? { iconId: record.iconId } : {}),
        ...(label ? { label } : {}),
      };

      return Object.keys(normalized).length > 0 ? [[resolvedPresetId, normalized]] : [];
    })
  );
};

const normalizeLayoutPresetOverrides = (overrides: unknown): LayoutPresetOverrides => {
  if (!overrides || typeof overrides !== 'object') {
    return {};
  }

  return Object.fromEntries(
    Object.entries(overrides as Record<string, unknown>).flatMap(([presetId, snapshot]) => {
      const resolvedPresetId = resolveLayoutPresetId(presetId);

      return isBuiltInLayoutPresetId(resolvedPresetId) && isLayoutPresetSnapshot(snapshot)
        ? [[resolvedPresetId, cloneLayoutPresetSnapshot(snapshot)]]
        : [];
    })
  );
};

export const normalizeWorkbenchAccount = (value: unknown): WorkbenchState['account'] => {
  const account = value && typeof value === 'object' ? (value as Partial<WorkbenchState['account']>) : undefined;
  const customLayoutPresets = normalizeCustomLayoutPresets(account?.customLayoutPresets);
  const resolvedActivePresetId = resolveLayoutPresetId(account?.activeLayoutPresetId ?? defaultLayoutPreset.id);
  const activeLayoutPresetId =
    isBuiltInLayoutPresetId(resolvedActivePresetId) ||
    customLayoutPresets.some((preset) => preset.id === resolvedActivePresetId)
      ? resolvedActivePresetId
      : defaultLayoutPreset.id;

  return {
    activeLayoutPresetId,
    customLayoutPresets,
    layoutPresetMetadataOverrides: normalizeLayoutPresetMetadataOverrides(account?.layoutPresetMetadataOverrides),
    layoutPresetOrder: normalizeLayoutPresetOrder(account?.layoutPresetOrder, [
      ...layoutPresets,
      ...customLayoutPresets,
    ]),
    layoutPresetOverrides: normalizeLayoutPresetOverrides(account?.layoutPresetOverrides),
    layoutPresetRouteOverrides: normalizeLayoutPresetRouteOverrides(account?.layoutPresetRouteOverrides),
  };
};

const normalizeWorkbenchState = (state: WorkbenchState): WorkbenchState => {
  // Built explicitly: legacy snapshots carried preferences inside the account
  // (they live in the settings store now) and must not resurface here.
  const account = normalizeWorkbenchAccount(state.account);
  const restored = state.projects.map((project) =>
    withoutRedundantWorkingCopies(account, normalizeWorkbenchProject(project))
  );
  // Every hydrated editor needs an active project, including projectless cached snapshots returned after canvas
  // recovery fails; seed a draft here for all load paths.
  const projects = restored.length > 0 ? restored : [createDraftProject([], account)];
  const activeProjectId = projects.some((project) => project.id === state.activeProjectId)
    ? state.activeProjectId
    : projects[0]!.id;

  return {
    ...state,
    account,
    activeProjectId,
    backendConnection: { status: 'connecting' },
    notifications: [],
    projects,
  };
};

const updateActiveLayout = (
  state: WorkbenchState,
  getLayout: (layout: ProjectLayoutState) => ProjectLayoutState
): WorkbenchState =>
  updateActiveProject(state, (project) => {
    const nextProject = pushUndo(project, 'Update layout');

    return {
      ...nextProject,
      events: prependProjectEvent(nextProject.events, {
        createdAt: now(),
        id: createId('event'),
        summary: 'Updated active layout',
        type: 'layout-updated',
      }),
      layout: getLayout(project.layout),
    };
  });

const getAvailableLayoutPreset = (state: WorkbenchState, presetId: LayoutPresetId): LayoutPreset =>
  resolveSavedLayoutPreset(state.account, presetId);

const setBuiltInLayoutPresetMetadata = (
  state: WorkbenchState,
  presetId: BuiltInLayoutPresetId,
  metadata: Required<LayoutPresetMetadataOverride>
): WorkbenchState => {
  const shippedPreset = getLayoutPreset(presetId);
  const override: LayoutPresetMetadataOverride = {
    ...(metadata.iconId !== shippedPreset.iconId ? { iconId: metadata.iconId } : {}),
    ...(metadata.label !== shippedPreset.label ? { label: metadata.label } : {}),
  };
  const layoutPresetMetadataOverrides: LayoutPresetMetadataOverrides = {
    ...state.account.layoutPresetMetadataOverrides,
  };

  if (Object.keys(override).length > 0) {
    layoutPresetMetadataOverrides[presetId] = override;
  } else {
    delete layoutPresetMetadataOverrides[presetId];
  }

  return { ...state, account: { ...state.account, layoutPresetMetadataOverrides } };
};

const applyLayoutPresetToProject = (project: Project, preset: LayoutPreset): Project => {
  // Presets replace all placements, including floating windows, and bypass normalizeWorkbenchProject.
  const snapshot = normalizeLayoutPresetSnapshot(preset.snapshot);
  const widgetInstances = { ...project.widgetInstances };

  for (const instance of Object.values(snapshot.widgetInstances)) {
    widgetInstances[instance.id] = widgetInstances[instance.id]
      ? { ...widgetInstances[instance.id], title: instance.title }
      : createWidgetInstance(instance.typeId, instance.id);
  }

  return {
    ...project,
    floatingWidgets: snapshot.floatingWidgets,
    // Presets carry no memory of their own, but a window the preset closes is still remembered.
    lastFloatingGeometry: rememberClosedWindows(
      project.lastFloatingGeometry,
      project.floatingWidgets,
      snapshot.floatingWidgets
    ),
    layout: {
      ...snapshot.layout,
      panels: { ...snapshot.layout.panels },
      presetId: preset.id,
    },
    widgetInstances,
    widgetRegions: cloneLayoutPresetWidgetRegions(snapshot.widgetRegions),
  };
};

const updateActiveProjectLayoutPreset = (
  state: WorkbenchState,
  preset: LayoutPreset,
  { applyDefaultRoute }: { applyDefaultRoute: boolean }
): WorkbenchState =>
  updateActiveProject(state, (project) => {
    const nextProject = pushUndo(project, 'Update layout');
    const nextLayoutProject = applyLayoutPresetToProject(nextProject, preset);

    return {
      ...nextLayoutProject,
      events: prependProjectEvent(nextProject.events, {
        createdAt: now(),
        id: createId('event'),
        summary: 'Updated active layout',
        type: 'layout-updated',
      }),
      invocation: applyDefaultRoute
        ? getInvocationAfterLayoutPreset(nextProject.invocation, preset)
        : nextLayoutProject.invocation,
    };
  });

/** Leave the active preset for another, keeping each preset's working arrangement in this project. */
const switchActiveProjectLayoutPreset = (state: WorkbenchState, preset: LayoutPreset): WorkbenchState =>
  updateActiveProject(state, (project) => {
    const nextProject = pushUndo(project, 'Update layout');
    const workingArrangement = findLayoutPresetWorkingCopy(project.presetWorkingLayouts, preset.id);
    // The working copy takes the saved preset's place for the arrangement only; identity and route stay the preset's.
    const nextLayoutProject = applyLayoutPresetToProject(
      nextProject,
      workingArrangement ? { ...preset, snapshot: workingArrangement } : preset
    );

    return {
      ...nextLayoutProject,
      events: prependProjectEvent(nextProject.events, {
        createdAt: now(),
        id: createId('event'),
        summary: 'Updated active layout',
        type: 'layout-updated',
      }),
      invocation: getInvocationAfterLayoutPreset(nextProject.invocation, preset),
      presetWorkingLayouts: captureOutgoingWorkingLayout(state.account, project, preset.id),
    };
  });

const resetActiveLayoutToSaved = (state: WorkbenchState): WorkbenchState => {
  const preset = getAvailableLayoutPreset(
    state,
    state.projects.find((project) => project.id === state.activeProjectId)?.layout.presetId ??
      state.account.activeLayoutPresetId
  );

  return updateActiveProjectLayoutPreset(state, preset, { applyDefaultRoute: false });
};

const updateActiveInvocation = (
  state: WorkbenchState,
  getInvocation: (invocation: InvocationRoute) => InvocationRoute
): WorkbenchState =>
  updateActiveProject(state, (project) => {
    const nextProject = pushUndo(project, 'Update invocation route');

    return {
      ...nextProject,
      events: prependProjectEvent(nextProject.events, {
        createdAt: now(),
        id: createId('event'),
        summary: 'Updated invocation source or destination',
        type: 'invocation-updated',
      }),
      invocation: getInvocation(project.invocation),
    };
  });

/** Auto-routing shares the triggering edit's update and creates no separate undo entry or event. */
const applyAutoRouteForEdit = (
  project: Project,
  sourceId: InvocationSourceId,
  context: WorkbenchReducerContext
): Project => {
  if (!context.autoSwitchInvocationRoute) {
    return project;
  }

  const invocation = getRouteAfterHighConfidenceEdit(project.invocation, sourceId);

  return invocation === project.invocation ? project : { ...project, invocation };
};

/** Generate also supplies Canvas parameters and dimensions, so its edits must not steal an active Canvas route. */
const applyAutoRouteForGenerateEdit = (project: Project, context: WorkbenchReducerContext): Project =>
  project.invocation.sourceId === 'canvas' ? project : applyAutoRouteForEdit(project, 'generate', context);

/**
 * Revealing a graph widget expresses invocation intent through the same lock/preference gates as editing; Generate
 * retains the Canvas exception.
 */
const applyAutoRouteForWidgetReveal = (
  project: Project,
  typeId: WidgetTypeId | undefined,
  context: WorkbenchReducerContext
): Project => {
  const sourceId = typeId ? getSourceIdForWidgetTypeId(typeId) : null;

  if (!sourceId) {
    return project;
  }

  return sourceId === 'generate'
    ? applyAutoRouteForGenerateEdit(project, context)
    : applyAutoRouteForEdit(project, sourceId, context);
};

/** `applyAutoRouteForWidgetReveal` for the instance-addressed reveal actions. */
const applyAutoRouteForRevealedInstance = (
  project: Project,
  instanceId: WidgetInstanceId,
  context: WorkbenchReducerContext
): Project => applyAutoRouteForWidgetReveal(project, project.widgetInstances[instanceId]?.typeId, context);

/**
 * Consequential tab changes route only when a different widget becomes visible; background closes, collapsed
 * regions, and dangling pointers reveal nothing.
 */
const applyAutoRouteForRegionFront = (
  project: Project,
  previousRegion: WidgetRegionState,
  region: WidgetRegion,
  context: WorkbenchReducerContext
): Project => {
  const nextRegion = project.widgetRegions[region];

  if (
    nextRegion.isCollapsed ||
    nextRegion.activeInstanceId === previousRegion.activeInstanceId ||
    !nextRegion.instanceIds.includes(nextRegion.activeInstanceId)
  ) {
    return project;
  }

  return applyAutoRouteForRevealedInstance(project, nextRegion.activeInstanceId, context);
};

const compileInvocationSnapshot = (
  project: Project,
  route: InvocationRoute,
  models?: readonly ModelConfig[],
  workflowGenerators?: WorkflowGeneratorResolutions,
  workflowDocument: ProjectGraphState = getActiveProjectGraph(project)
): {
  graph: GraphContract;
  widgetStates: WidgetStateMap;
  workflowJson?: Record<string, unknown>;
  workflow?: Omit<WorkflowSubmissionPlan, 'graph'>;
  /** The project workflow the graph was compiled from; results and seed advances return to it. */
  projectWorkflowId?: string;
} | null => {
  const widgetStates = getWidgetStatesSnapshot(project.widgetInstances);
  const randDevice = resolveRandDeviceMetadata(project.settings.useCpuNoise, getGenerationDevicesSnapshot().options);

  if (route.sourceId === 'workflow') {
    // Route validation guarantees templates are loaded before this imperative read.
    const templatesSnapshot = getInvocationTemplatesSnapshot();

    if (templatesSnapshot.status !== 'loaded') {
      return null;
    }

    const plan = planWorkflowSubmission(workflowDocument, templatesSnapshot.templates, {
      autoBoardId: getGalleryAutoAddBoardId(widgetStates.gallery?.values ?? {}),
      batchCount: sanitizeBatchCount(widgetStates.workflow?.values.batchCount),
      generators: workflowGenerators,
    });

    // Null means a generator is still unresolved or was edited during its round trip; nothing is queued.
    if (!plan) {
      return null;
    }

    const { graph, ...workflow } = plan;
    const { id: _id, ...workflowJson } = serializeWorkflowJsonForSubmission(workflowDocument);

    return {
      graph,
      projectWorkflowId: workflowDocument.id,
      widgetStates,
      workflow,
      ...(hasMultipleWorkflowReturnNodes(workflowDocument) ? {} : { workflowJson }),
    };
  }

  if (route.sourceId === 'upscale') {
    const values = normalizeUpscaleWidgetValues(getWidgetValues(project, 'upscale'));

    if (!values) {
      return null;
    }

    const syncedValues = models ? syncUpscaleWidgetValuesWithModels(values, models) : values;
    const currentValues: UpscaleWidgetValues = {
      ...syncedValues,
      ...getPromptDraftFromValues(getProjectWidgetValues(project, 'generate')),
    };

    if (getUpscaleValidationReasons(currentValues, models).length > 0) {
      return null;
    }

    const resolvedValues: UpscaleWidgetValues = { ...currentValues, seed: resolveUpscaleSeed(currentValues) };
    const compiledGraph = compileUpscaleGraph(resolvedValues, route.destination, project.settings, randDevice).graph;

    widgetStates.upscale = {
      ...widgetStates.upscale,
      graphId: compiledGraph.id,
      values: { ...cloneUpscaleWidgetValues(resolvedValues) },
    };

    return { graph: compiledGraph, widgetStates };
  }

  if (route.sourceId === 'video') {
    const values = normalizeVideoWidgetValues(getWidgetValues(project, 'video'));

    if (!values) {
      return null;
    }

    // Video's prompt is its own widget value, deliberately independent of the
    // draft Generate and Upscale share: no draft is merged in here.
    const currentValues: VideoWidgetValues = models ? syncVideoWidgetValuesWithModels(values, models) : values;

    if (!currentValues.model || getVideoWidgetValidationReasons(currentValues, models).length > 0) {
      return null;
    }

    const resolvedValues: VideoWidgetValues = { ...currentValues, seed: resolveVideoSeed(currentValues) };
    const compiledGraph = compileVideoGraph(resolvedValues, currentValues.model).graph;

    widgetStates.video = {
      ...widgetStates.video,
      graphId: compiledGraph.id,
      values: { ...cloneVideoWidgetValues(resolvedValues) },
    };

    return { graph: compiledGraph, widgetStates };
  }

  if (route.sourceId !== 'generate') {
    const widgetGraph = project.widgetGraphs[route.sourceId as WidgetTypeId];

    return widgetGraph ? { graph: cloneGraph(widgetGraph), widgetStates } : null;
  }

  const values = normalizeGenerateWidgetValues(getWidgetValues(project, 'generate'));

  if (!values) {
    return null;
  }

  const currentValues = models ? syncGenerateWidgetValuesWithModels(values, models) : values;
  const availabilityReasons = models
    ? getGenerationModelAvailabilityReasons(currentValues.model, currentValues, models)
    : [];

  if (availabilityReasons.length > 0) {
    return null;
  }

  const resolvedSettings: GenerateWidgetValues = {
    ...currentValues,
    seed: resolveGenerateSeed(currentValues),
  };
  const compiledGraph = compileGenerateGraph(
    resolvedSettings,
    resolvedSettings.model,
    route.destination,
    project.settings,
    randDevice
  ).graph;

  widgetStates.generate = {
    ...widgetStates.generate,
    graphId: compiledGraph.id,
    values: cloneGenerateWidgetValues(resolvedSettings),
  };

  return { graph: compiledGraph, widgetStates };
};

const updateProjectById = (
  state: WorkbenchState,
  projectId: string,
  getProject: (project: Project) => Project
): WorkbenchState => {
  let didChange = false;
  const projects = state.projects.map((project) => {
    if (project.id !== projectId) {
      return project;
    }

    const nextProject = getProject(project);

    if (nextProject !== project) {
      didChange = true;
    }

    return nextProject;
  });

  return didChange ? { ...state, projects } : state;
};

const updateGalleryValues = (
  state: WorkbenchState,
  getValues: (values: Record<string, unknown>) => Record<string, unknown>,
  projectId = state.activeProjectId
): WorkbenchState => {
  const targetProject = state.projects.find((project) => project.id === projectId);
  const values = targetProject ? getWidgetValues(targetProject, 'gallery') : null;

  if (!targetProject || !values) {
    return state;
  }

  const nextValues = getValues(values);

  if (nextValues === values) {
    return state;
  }

  return updateProjectById(state, projectId, (project) =>
    updateProjectWidgetValues(project, 'gallery', () => nextValues)
  );
};

const updateAllProjectGalleryValues = (
  state: WorkbenchState,
  getValues: (values: Record<string, unknown>) => Record<string, unknown>
): WorkbenchState => {
  let didChange = false;
  const projects = state.projects.map((project) => {
    const nextProject = updateProjectWidgetValues(project, 'gallery', getValues);

    didChange ||= nextProject !== project;
    return nextProject;
  });

  return didChange ? { ...state, projects } : state;
};

const patchGalleryItemsAcrossProjects = (
  state: WorkbenchState,
  itemKeys: ReadonlySet<GalleryItemKey>,
  changes: Partial<Pick<GalleryItem, 'boardId' | 'starred'>>
): WorkbenchState => {
  if (itemKeys.size === 0) {
    return state;
  }

  return updateAllProjectGalleryValues(state, (values) => {
    let didChange = false;
    const patchPersistedItem = (value: unknown, imageOnly = false): unknown => {
      const item = getGalleryItemFromPersistedValue(values, value);

      if (!item || (imageOnly && item.kind !== 'image') || !itemKeys.has(toGalleryItemKey(item))) {
        return value;
      }

      didChange = true;
      return { ...item, ...changes };
    };
    const recentImages = getGalleryImages(values).map((image) => {
      if (!itemKeys.has(toGalleryItemKey({ kind: 'image', name: image.imageName }))) {
        return image;
      }

      didChange = true;
      return { ...image, ...changes };
    });
    const selectedImage = patchPersistedItem(values.selectedImage);
    const compareImage = patchPersistedItem(values.compareImage, true);
    const selectedImageMoved = changes.boardId !== undefined && selectedImage !== values.selectedImage;
    const selectedImageQuery =
      selectedImageMoved && values.selectedImageQuery && typeof values.selectedImageQuery === 'object'
        ? {
            ...(values.selectedImageQuery as Record<string, unknown>),
            boardId: changes.boardId,
            page: 0,
            paginationMode: getGallerySettings(values).paginationMode,
            searchTerm: '',
          }
        : values.selectedImageQuery;

    return didChange
      ? {
          ...values,
          compareImage,
          recentImages,
          selectedImage,
          ...(selectedImageMoved ? { selectedImagePage: 0, selectedImageQuery } : {}),
        }
      : values;
  });
};

const getLocallyKnownGalleryItemsOnBoard = (
  state: WorkbenchState,
  boardId: string
): Map<GalleryItemKey, GalleryItem> => {
  const items = new Map<GalleryItemKey, GalleryItem>();

  for (const project of state.projects) {
    const values = getWidgetValues(project, 'gallery');
    const candidates: GalleryItem[] = [
      ...getGalleryImages(values).map((image) => legacyGeneratedImageToGalleryItem(image)),
      ...[values.selectedImage, values.compareImage].flatMap((value) => {
        const item = getGalleryItemFromPersistedValue(values, value);

        return item ? [item] : [];
      }),
    ];

    for (const item of candidates) {
      if (item.boardId === boardId) {
        items.set(toGalleryItemKey(item), item);
      }
    }
  }

  return items;
};

const removeGalleryItemsFromAllProjects = (
  state: WorkbenchState,
  removedItemKeys: ReadonlySet<GalleryItemKey>
): WorkbenchState => {
  if (removedItemKeys.size === 0) {
    return state;
  }

  const removedImageNames = new Set(
    [...removedItemKeys].flatMap((key) => {
      const ref = parseGalleryItemKey(key);

      return ref.kind === 'image' ? [ref.name] : [];
    })
  );
  const removedVideoNames = new Set(
    [...removedItemKeys].flatMap((key) => {
      const ref = parseGalleryItemKey(key);

      return ref.kind === 'video' ? [ref.name] : [];
    })
  );
  let didChange = false;
  const projects = state.projects.map((project) => {
    const withoutGalleryItems = updateProjectWidgetValues(project, 'gallery', (values) => {
      const selectedImage = values.selectedImage;
      const compareImage = values.compareImage;
      const selectedImageName = typeof values.selectedImageName === 'string' ? values.selectedImageName : null;
      const recentImages = getGalleryImages(values);
      const selectedItemKeys = getPersistedSelectedGalleryItemKeys(values);
      const selectedItem = getGalleryItemFromPersistedValue(values, selectedImage);
      const compareItem = getGalleryItemFromPersistedValue(values, compareImage);
      const selectedImageKey = selectedItem ? toGalleryItemKey(selectedItem) : null;
      const compareImageKey = compareItem ? toGalleryItemKey(compareItem) : null;
      const selectedNameKey = selectedImageName ? canonicalizeGalleryItemKey(selectedImageName) : null;
      const nextRecentImages = recentImages.filter(
        (image) => !removedItemKeys.has(toGalleryItemKey({ kind: 'image', name: image.imageName }))
      );
      const nextSelectedItemKeys = selectedItemKeys.filter((key) => !removedItemKeys.has(key));
      const nextSelectedImage = selectedImageKey && removedItemKeys.has(selectedImageKey) ? null : selectedImage;
      const nextCompareImage = compareImageKey && removedItemKeys.has(compareImageKey) ? null : compareImage;
      const nextSelectedImageName = selectedNameKey && removedItemKeys.has(selectedNameKey) ? null : selectedImageName;

      if (
        nextRecentImages.length === recentImages.length &&
        nextSelectedItemKeys.length === selectedItemKeys.length &&
        nextSelectedImage === selectedImage &&
        nextCompareImage === compareImage &&
        nextSelectedImageName === selectedImageName
      ) {
        return values;
      }

      return {
        ...values,
        compareImage: nextCompareImage,
        recentImages: nextRecentImages,
        selectedImage: nextSelectedImage,
        selectedImageName: nextSelectedImageName,
        selectedImageNames: nextSelectedItemKeys,
      };
    });
    const withoutUpscaleInput = updateProjectWidgetValues(withoutGalleryItems, 'upscale', (rawValues) => {
      const values = normalizeUpscaleWidgetValues(rawValues);

      if (!values?.inputImage || !removedImageNames.has(values.inputImage.image_name)) {
        return rawValues;
      }

      return { ...clearDeletedUpscaleInput(values, removedImageNames) };
    });

    // Sweep raw slots before normalization so mutually excluded media references cannot survive deletion and
    // reappear later.
    const withoutVideoMedia = updateProjectWidgetValues(withoutUpscaleInput, 'video', (rawValues) =>
      clearDeletedVideoMedia(rawValues, removedImageNames, removedVideoNames)
    );

    didChange ||= withoutVideoMedia !== project;
    return withoutVideoMedia;
  });

  return didChange ? { ...state, projects } : state;
};

const reconcileDeletedGalleryBoard = (
  state: WorkbenchState,
  boardId: string,
  deletedItemKeys: ReadonlySet<GalleryItemKey>,
  confirmedMovedItemKeys: ReadonlySet<GalleryItemKey>
): WorkbenchState => {
  const survivingItemKeys = new Set(
    [...getLocallyKnownGalleryItemsOnBoard(state, boardId).keys(), ...confirmedMovedItemKeys].filter(
      (key) => !deletedItemKeys.has(key)
    )
  );
  const withoutDeletedItems = removeGalleryItemsFromAllProjects(state, deletedItemKeys);
  const withSurvivorsMoved = patchGalleryItemsAcrossProjects(withoutDeletedItems, survivingItemKeys, {
    boardId: 'none',
  });
  const withBoardReferencesCleared = updateAllProjectGalleryValues(withSurvivorsMoved, (values) => {
    const selectedBoardWasDeleted = values.selectedBoardId === boardId;
    const projectBoardWasDeleted = values.projectBoardId === boardId;
    const autoAddBoardWasDeleted = values.autoAddBoardId === boardId;

    if (!selectedBoardWasDeleted && !projectBoardWasDeleted && !autoAddBoardWasDeleted) {
      return values;
    }

    return {
      ...values,
      ...(selectedBoardWasDeleted
        ? { galleryPage: 0, selectedBoardId: 'none', semanticImageQuery: null, semanticSearchText: null }
        : {}),
      ...(projectBoardWasDeleted ? { projectBoardId: null } : {}),
      ...(autoAddBoardWasDeleted ? { autoAddBoardId: GALLERY_AUTO_ADD_FOLLOW } : {}),
    };
  });
  let didChangeQueue = false;
  const projects = withBoardReferencesCleared.projects.map((project) => {
    let didChangeItems = false;
    const items = project.queue.items.map((item) => {
      if ((item.status !== 'pending' && item.status !== 'running') || item.snapshot.galleryBoardId !== boardId) {
        return item;
      }

      didChangeItems = true;
      return { ...item, snapshot: { ...item.snapshot, galleryBoardId: 'none' } };
    });

    if (!didChangeItems) {
      return project;
    }

    didChangeQueue = true;
    return { ...project, queue: { ...project.queue, items } };
  });

  return didChangeQueue ? { ...withBoardReferencesCleared, projects } : withBoardReferencesCleared;
};

/** A deliberate selection pauses live-follow; only generations submitted after that selection may take the preview. */
const updateGalleryValuesAndPauseLiveFollow = (
  state: WorkbenchState,
  getValues: (values: Record<string, unknown>) => Record<string, unknown>,
  projectId = state.activeProjectId
): WorkbenchState =>
  updateProjectById(state, projectId, (project) =>
    updateProjectWidgetValues(
      {
        ...project,
        settings: { ...project.settings, showProgressImagesInViewer: false },
      },
      'gallery',
      (values) => ({ ...getValues(values), liveFollowPausedAt: now() })
    )
  );

const getLiveFollowPausedAt = (project: Project): string | null => {
  const pausedAt = getWidgetValues(project, 'gallery').liveFollowPausedAt;

  return typeof pausedAt === 'string' ? pausedAt : null;
};

/** Submission resumes selection-paused live-follow, but preserves an explicit opt-out. */
const shouldResumeLiveFollowOnSubmit = (project: Project): boolean =>
  !project.settings.showProgressImagesInViewer && getLiveFollowPausedAt(project) !== null;

/** Only generations submitted after the deliberate selection may replace it. */
const isSubmittedAfterLiveFollowPause = (project: Project, sourceQueueItemId: string): boolean => {
  const pausedAt = getLiveFollowPausedAt(project);

  if (pausedAt === null) {
    return true;
  }

  const submittedAt = project.queue.items.find((item) => item.id === sourceQueueItemId)?.snapshot.submittedAt;

  return submittedAt !== undefined && submittedAt > pausedAt;
};

const updateQueueItem = (project: Project, queueItemId: string, getItem: (item: QueueItem) => QueueItem): Project => {
  let didChange = false;
  const items = project.queue.items.map((item) => {
    if (item.id !== queueItemId) {
      return item;
    }

    const nextItem = getItem(item);

    if (nextItem !== item) {
      didChange = true;
    }

    return nextItem;
  });

  return didChange ? { ...project, queue: { items } } : project;
};

const isClearableQueueItem = (item: QueueItem): boolean => item.status === 'completed' || item.status === 'failed';

const shouldApplyQueueBulkActionToProject = (project: Project, projectId?: string): boolean =>
  projectId === undefined || project.id === projectId;

const mergeImageResults = (
  existingImages: GeneratedImageContract[] | undefined,
  incomingImages: GeneratedImageContract[]
): GeneratedImageContract[] => {
  const existing = existingImages ?? [];
  const existingNames = new Set(existing.map((image) => image.imageName));

  return [...existing, ...incomingImages.filter((image) => !existingNames.has(image.imageName))];
};

const mergeBackendItemId = (ids: number[] | undefined, backendItemId: number): number[] =>
  ids?.includes(backendItemId) ? ids : [...(ids ?? []), backendItemId];

const getQueueItemStatusAfterBackendCancellation = (
  item: QueueItem,
  cancelledBackendItemIds: number[]
): QueueHistoryItemStatus => {
  if (!item.backendItemIds?.length) {
    return item.status;
  }

  const completedBackendItemIds = new Set(item.completedBackendItemIds ?? []);
  const terminalBackendItemIds = new Set([...completedBackendItemIds, ...cancelledBackendItemIds]);
  const isEveryBackendItemTerminal = item.backendItemIds.every((backendItemId) =>
    terminalBackendItemIds.has(backendItemId)
  );

  if (!isEveryBackendItemTerminal) {
    return item.status;
  }

  return completedBackendItemIds.size > 0 || (item.resultImages?.length ?? 0) > 0 ? 'completed' : 'cancelled';
};

/** Whether a newly routed result may take the Gallery selection (and so the settled Preview). */
const shouldSelectGalleryResult = (
  project: Project,
  galleryValues: Record<string, unknown>,
  sourceQueueItemId: string
): boolean =>
  typeof galleryValues.selectedImageName !== 'string' ||
  (project.settings.showProgressImagesInViewer && isSubmittedAfterLiveFollowPause(project, sourceQueueItemId));

const getGalleryResultSelectionValues = (
  galleryValues: Record<string, unknown>,
  item: GalleryItem
): Record<string, unknown> => {
  const itemKey = toGalleryItemKey(item);
  const gallerySettings = getGallerySettings(galleryValues);

  return {
    ...(item.kind === 'video' ? { compareImage: null } : {}),
    selectedImage: item,
    selectedImageName: itemKey,
    selectedImageNames: [itemKey],
    selectedImagePage: 0,
    selectedImageQuery: {
      boardId: item.boardId,
      galleryView: item.category === 'general' ? 'images' : 'assets',
      imageOrderDir: gallerySettings.imageOrderDir,
      page: 0,
      paginationMode: gallerySettings.paginationMode,
      searchTerm: '',
      starredOnly: false,
    },
  };
};

const updateGalleryWithResultImages = (project: Project, images: GeneratedImageContract[]): Project => {
  if (images.length === 0) {
    return project;
  }

  const galleryValues = getWidgetValues(project, 'gallery');
  const previousImages = getGalleryImages(galleryValues);
  const previousImageNames = new Set(previousImages.map((image) => image.imageName));
  const queueBoardIds = new Map(
    project.queue.items.map((item) => [item.id, item.snapshot.galleryBoardId ?? 'none'] as const)
  );
  const incomingImages = getBoundedRecentImages([...images].reverse());
  const newImages: GalleryImage[] = incomingImages
    .filter((image) => !previousImageNames.has(image.imageName))
    .map((image) => normalizeGalleryImage(image, queueBoardIds.get(image.sourceQueueItemId)));
  const nextSelectedImage =
    newImages[0] && shouldSelectGalleryResult(project, galleryValues, newImages[0].sourceQueueItemId)
      ? newImages[0]
      : undefined;
  return updateProjectWidgetValues(project, 'gallery', () => ({
    ...galleryValues,
    recentImages: getBoundedRecentImages([...newImages, ...previousImages]),
    ...(nextSelectedImage
      ? getGalleryResultSelectionValues(galleryValues, legacyGeneratedImageToGalleryItem(nextSelectedImage))
      : { selectedImageNames: getPersistedSelectedGalleryItemKeys(galleryValues) }),
  }));
};

/**
 * Videos are not kept in recent images, so the queue item records which ones were routed: only the newest video not
 * routed before may take the selection, or a repeat pass would take it back from a later run.
 */
const routeGalleryResultVideos = (
  project: Project,
  queueItemId: string,
  videos: readonly GeneratedVideoContract[],
  /** The project before this routing pass selected any image, so a run's video outranks its own images. */
  policyProject: Project = project
): Project => {
  const routedNames = new Set(project.queue.items.find((item) => item.id === queueItemId)?.resultVideoNames);
  const newVideos = videos.filter((video) => !routedNames.has(video.videoName));
  const video = newVideos.at(-1);

  if (!video) {
    return project;
  }

  const nextProject = updateQueueItem(project, queueItemId, (item) => ({
    ...item,
    resultVideoNames: [...(item.resultVideoNames ?? []), ...newVideos.map((newVideo) => newVideo.videoName)],
  }));
  if (!shouldSelectGalleryResult(policyProject, getWidgetValues(policyProject, 'gallery'), video.sourceQueueItemId)) {
    return nextProject;
  }

  const galleryValues = getWidgetValues(nextProject, 'gallery');

  return updateProjectWidgetValues(nextProject, 'gallery', () => ({
    ...galleryValues,
    ...getGalleryResultSelectionValues(galleryValues, generatedVideoToGalleryItem(video)),
  }));
};

const routeQueueItemPartialResults = (
  project: Project,
  queueItemId: string,
  backendItemId: number,
  images: GeneratedImageContract[],
  videos: readonly GeneratedVideoContract[] = []
): Project => {
  const queueItem = project.queue.items.find((item) => item.id === queueItemId);
  const destination = queueItem?.snapshot.destination ?? project.invocation.destination;
  const previousSelectedSlot = getSelectedCanvasStagingSlot(project);
  const nextProject = updateQueueItem(project, queueItemId, (item) => ({
    ...item,
    completedBackendItemIds: item.completedBackendItemIds?.includes(backendItemId)
      ? item.completedBackendItemIds
      : [...(item.completedBackendItemIds ?? []), backendItemId],
    resultImages: mergeImageResults(item.resultImages, images),
  }));

  if (destination === 'gallery') {
    // A run's video is its primary output; it takes the selection over any images routed alongside it.
    return routeGalleryResultVideos(
      updateGalleryWithResultImages(nextProject, images),
      queueItemId,
      videos,
      nextProject
    );
  }

  if (images.length === 0) {
    return clampCanvasStagingSelection(nextProject, previousSelectedSlot);
  }

  return clampCanvasStagingSelection(
    stageCanvasResultImages(
      nextProject,
      queueItemId,
      images,
      images.map(() => backendItemId),
      previousSelectedSlot
    )
  );
};

/** Seed advances are bookkeeping on the submitted workflow, not an undoable edit. */
const advanceWorkflowSeeds = (
  project: Project,
  workflowId: string,
  advances: WorkflowSubmissionPlan['seedAdvances']
): Project => {
  const entry = findProjectWorkflow(project, workflowId);

  return entry
    ? setProjectWorkflowDocument(
        project,
        workflowId,
        projectGraphReducer(entry.document, { advances, type: 'advanceSeedFields' })
      )
    : project;
};

/** A completed workflow run becomes the originating project workflow's preview; other copies are untouched. */
const recordWorkflowRunPreview = (
  project: Project,
  queueItem: QueueItem | undefined,
  images: GeneratedImageContract[]
): Project => {
  const submission = queueItem?.snapshot.backendSubmission;
  const lastImage = images.at(-1);

  if (!queueItem || submission?.kind !== 'workflow' || !submission.projectWorkflowId || !lastImage) {
    return project;
  }

  return recordProjectWorkflowRun(project, submission.projectWorkflowId, {
    completedAt: now(),
    imageName: lastImage.imageName,
    submittedAt: queueItem.snapshot.submittedAt,
  });
};

const routeQueueItemResults = (
  sourceProject: Project,
  queueItemId: string,
  images: GeneratedImageContract[],
  videos: readonly GeneratedVideoContract[] = []
): Project => {
  const queueItem = sourceProject.queue.items.find((item) => item.id === queueItemId);
  const project = recordWorkflowRunPreview(sourceProject, queueItem, images);
  const destination = queueItem?.snapshot.destination ?? project.invocation.destination;
  const previousSelectedSlot = getSelectedCanvasStagingSlot(project);
  const nextProject = updateQueueItem(project, queueItemId, (item) => ({
    ...item,
    completedBackendItemIds: item.backendItemIds
      ? item.backendItemIds.filter((backendItemId) => !item.cancelledBackendItemIds?.includes(backendItemId))
      : item.completedBackendItemIds,
    resultImages: images,
    status: 'completed',
  }));

  if (destination === 'gallery') {
    // A run's video is its primary output; it takes the selection over any images routed alongside it.
    return routeGalleryResultVideos(
      updateGalleryWithResultImages(nextProject, images),
      queueItemId,
      videos,
      nextProject
    );
  }

  // Stage results only into the submitted documentRevision; wholesale canvas swaps invalidate staging while
  // preserving completion status.
  if (queueItem && queueItem.snapshot.canvas.documentRevision !== nextProject.canvas.documentRevision) {
    return nextProject;
  }

  if (images.length === 0) {
    return clampCanvasStagingSelection(nextProject, previousSelectedSlot);
  }

  const sourceBackendItemIds = queueItem?.backendItemIds?.filter(
    (backendItemId) => !queueItem.cancelledBackendItemIds?.includes(backendItemId)
  );

  return stageCanvasResultImages(nextProject, queueItemId, images, sourceBackendItemIds, previousSelectedSlot);
};

/** Enqueue compiled snapshots from route validation or asynchronous canvas preparation. */
const enqueueCompiledSnapshot = (
  project: Project,
  route: InvocationRoute,
  compiled: Partial<ExpandedPositivePrompts> & {
    generate?: QueueGenerateSnapshot;
    graph: GraphContract;
    widgetStates: WidgetStateMap;
    /** The serialized parent workflow, without its library record id. */
    workflowJson?: Record<string, unknown>;
    /** The workflow route's seed plan: batch data, run count, and the fields to advance. */
    workflow?: Omit<WorkflowSubmissionPlan, 'graph'>;
    /** The project workflow the graph was compiled from; results and seed advances return to it. */
    projectWorkflowId?: string;
  },
  backendSupportsCancellation: boolean,
  canvasSnapshot?: CanvasStateContractV3
): Project => {
  const submittedAt = now();
  const queueItemId = createId('queue-item');
  const { generate, graph } = compiled;
  const widgetStates = Object.fromEntries(
    Object.entries(applyQueueGenerateSnapshotToWidgetStates(compiled.widgetStates, generate)).map(
      ([typeId, widgetState]) => [typeId, cloneWidgetState(widgetState)]
    )
  ) as WidgetStateMap;
  const generateSettings =
    route.sourceId === 'generate' ? normalizeGenerateSettings(widgetStates.generate.values) : null;
  const upscaleSettings =
    route.sourceId === 'upscale' ? normalizeUpscaleWidgetValues(widgetStates.upscale.values) : null;
  const videoSettings = route.sourceId === 'video' ? normalizeVideoWidgetValues(widgetStates.video.values) : null;
  const backendGraph = graph.backendGraph;
  const canvasGenerateSettings = route.sourceId === 'canvas' ? normalizeGenerateSettings(generate?.values) : null;
  const sourceGenerateSettings =
    route.sourceId === 'canvas'
      ? canvasGenerateSettings
      : route.sourceId === 'generate'
        ? generateSettings
        : route.sourceId === 'upscale'
          ? upscaleSettings
          : route.sourceId === 'video'
            ? videoSettings
            : null;
  // Resolve template-merged prompts here, where Generate and Canvas submissions converge; Upscale merges to
  // identity.
  const effectivePrompts = sourceGenerateSettings ? getEffectivePrompts(sourceGenerateSettings) : null;
  // Record expansions for Generate routes even when only one prompt results. Inspect merged prompts because
  // templates may introduce dynamic syntax.
  const expandedPositivePrompts =
    route.sourceId !== 'upscale' &&
    route.sourceId !== 'video' &&
    compiled.positivePrompts &&
    compiled.positivePrompts.length > 0 &&
    effectivePrompts &&
    hasDynamicPromptSyntax(effectivePrompts.positivePrompt)
      ? compiled.positivePrompts
      : undefined;
  const expandedSeedBehaviour = expandedPositivePrompts
    ? (canvasGenerateSettings ?? generateSettings)?.dynamicPromptsSeedBehaviour
    : undefined;
  // The draw the expansion used. Generate expands before dispatch, so the project's sample seed may have rotated
  // since; recall and rotation follow the expansion, not the project.
  const expansionSampleSeed = expandedPositivePrompts ? (compiled.positivePromptsSampleSeed ?? null) : null;
  // Compilation already reserved the starting seed; failures and cancellations do not roll the sequence back.
  const seedPlan = sourceGenerateSettings
    ? planSeedSubmission({
        batchCount: sourceGenerateSettings.batchCount,
        promptCount: expandedPositivePrompts?.length ?? 1,
        seedBehaviour: expandedSeedBehaviour ?? 'per-iteration',
        seedMode: sourceGenerateSettings.seedMode,
        startSeed: sourceGenerateSettings.seed,
      })
    : null;
  const backendSubmission: QueueCompiledSubmission = !backendGraph
    ? { error: `${route.sourceId} queue item is missing a compiled backend graph.`, kind: 'invalid' }
    : route.sourceId === 'workflow'
      ? !compiled.workflow
        ? { error: 'workflow queue item is missing its seed plan.', kind: 'invalid' }
        : {
            batchCount: compiled.workflow.batchCount,
            ...(compiled.workflow.seeds.length ? { seeds: compiled.workflow.seeds } : {}),
            ...(compiled.workflow.batchData.length ? { batchData: compiled.workflow.batchData } : {}),
            graph: backendGraph,
            kind: 'workflow',
            ...(compiled.workflowJson ? { workflow: compiled.workflowJson } : {}),
            // Capture the originating project workflow so results and seed advances return to it even after the
            // active workflow changes.
            ...(compiled.projectWorkflowId ? { projectWorkflowId: compiled.projectWorkflowId } : {}),
          }
      : sourceGenerateSettings && effectivePrompts
        ? {
            batchCount: sourceGenerateSettings.batchCount,
            graph: backendGraph,
            kind: 'generate',
            // Disabling the negative prompt also suppresses the template's negative side.
            negativePrompt: sourceGenerateSettings.negativePromptEnabled ? effectivePrompts.negativePrompt : '',
            negativePromptNodeId: generate?.negativePromptNodeId ?? 'negative_prompt',
            positivePrompt: effectivePrompts.positivePrompt,
            positivePromptNodeId: generate?.positivePromptNodeId ?? 'positive_prompt',
            ...(expandedPositivePrompts ? { positivePrompts: expandedPositivePrompts } : {}),
            seed: sourceGenerateSettings.seed,
            ...(expandedSeedBehaviour ? { seedBehaviour: expandedSeedBehaviour } : {}),
            seedNodeId: generate?.seedNodeId ?? 'seed',
            seedStep: seedPlan?.step ?? 0,
          }
        : { error: `${route.sourceId} queue item is missing source submission metadata.`, kind: 'invalid' };
  // A workflow compiles every node's board in (Auto included), so its board-less results stay Uncategorized.
  const galleryBoardId =
    route.sourceId === 'workflow' ? null : getGalleryAutoAddBoardId(widgetStates.gallery?.values ?? {});
  const generatePresentationSettings = normalizeGenerateSettings(widgetStates.generate?.values);
  const videoPresentationDimensions =
    route.sourceId === 'video' && videoSettings?.model ? getVideoDimensions(videoSettings.model, videoSettings) : null;
  const presentationDimensions =
    route.sourceId === 'upscale' && upscaleSettings?.inputImage
      ? getUpscaleOutputDimensions(upscaleSettings.inputImage, upscaleSettings.scale)
      : videoPresentationDimensions
        ? videoPresentationDimensions
        : {
            height: generatePresentationSettings?.height ?? project.canvas.document.height,
            width: generatePresentationSettings?.width ?? project.canvas.document.width,
          };
  const submittedCanvas = canvasSnapshot ?? project.canvas;
  const generateRecallValues =
    route.sourceId === 'canvas'
      ? generate?.values
      : route.sourceId === 'generate'
        ? normalizeGenerateWidgetValues(widgetStates.generate?.values)
        : null;
  const recall = generateRecallValues
    ? {
        generateValues: {
          ...cloneGenerateWidgetValues(generateRecallValues),
          ...(expansionSampleSeed === null ? {} : { dynamicPromptsSampleSeed: expansionSampleSeed }),
        },
      }
    : videoSettings
      ? { videoValues: cloneVideoWidgetValues(videoSettings) }
      : undefined;
  const queueItem: QueueItem = {
    cancellable: backendSupportsCancellation,
    id: queueItemId,
    localRecoveryState: 'local-only',
    snapshot: {
      backendSubmission,
      canvas: {
        document: {
          bbox: { ...submittedCanvas.document.bbox },
          height: submittedCanvas.document.height,
          width: submittedCanvas.document.width,
        },
        documentRevision: submittedCanvas.documentRevision,
      },
      destination: route.destination,
      filterIntermediateResults: route.sourceId === 'workflow',
      galleryBoardId,
      graph: { id: graph.id, label: graph.label },
      presentation: {
        // Placeholder sizing only: superseded by the backend's real item ids as
        // soon as the batch is accepted.
        batchCount:
          backendSubmission.kind === 'invalid'
            ? 1
            : Math.min(
                MAX_QUEUE_BATCH_ITEMS,
                backendSubmission.batchCount *
                  (expandedPositivePrompts?.length ?? 1) *
                  (compiled.workflow?.batchSize ?? 1)
              ),
        height: presentationDimensions.height,
        // Use merged prompts so queue text stays consistent when the backend session arrives.
        ...(effectivePrompts?.positivePrompt ? { positivePrompt: effectivePrompts.positivePrompt } : {}),
        width: presentationDimensions.width,
      },
      sourceId: route.sourceId,
      ...(route.sourceId === 'generate' || route.sourceId === 'canvas'
        ? { resultNodeIds: ['canvas_output'] }
        : route.sourceId === 'upscale'
          ? { resultNodeIds: ['upscale_output'] }
          : route.sourceId === 'video'
            ? { resultNodeIds: ['video_output'] }
            : {}),
      submittedAt,
      ...(recall ? { recall } : {}),
    },
    status: 'pending',
  };

  // Advance seeds atomically with enqueueing; for async Canvas submissions, preserve any settings edited since
  // compilation.
  const advancedProject =
    seedPlan !== null && seedPlan.nextSeed !== null
      ? updateProjectWidgetValues(project, route.sourceId === 'canvas' ? 'generate' : route.sourceId, (values) =>
          values.seed === seedPlan.startSeed && values.seedMode === seedPlan.seedMode
            ? { ...values, seed: seedPlan.nextSeed }
            : values
        )
      : compiled.workflow && compiled.workflow.seedAdvances.length > 0 && compiled.projectWorkflowId
        ? advanceWorkflowSeeds(project, compiled.projectWorkflowId, compiled.workflow.seedAdvances)
        : project;
  // The queued item carries the previewed draw; rotating afterward makes the preview show the next one.
  const dynamicPromptsSettings = canvasGenerateSettings ?? generateSettings;
  // Fencing on the expansion's seed keeps a submission that drew an already-rotated sample from rotating again.
  const resampledProject =
    expansionSampleSeed !== null &&
    dynamicPromptsSettings &&
    dynamicPromptsSettings.dynamicPromptsCombinatorial === false &&
    dynamicPromptsSettings.dynamicPromptsResample
      ? updateProjectWidgetValues(
          advancedProject,
          route.sourceId === 'canvas' ? 'generate' : route.sourceId,
          (values) =>
            values.dynamicPromptsSampleSeed === expansionSampleSeed && values.dynamicPromptsCombinatorial === false
              ? { ...values, dynamicPromptsSampleSeed: createDynamicPromptsSampleSeed() }
              : values
        )
      : advancedProject;

  return {
    ...resampledProject,
    events: prependProjectEvent(project.events, {
      createdAt: submittedAt,
      id: createId('event'),
      runId: queueItemId,
      summary: `Submitted immutable ${route.sourceId} graph snapshot to ${route.destination}`,
      type: 'queue-submitted',
    }),
    promptHistory: generateSettings
      ? addPromptHistoryItem(project.promptHistory, getPromptHistoryItemFromGenerateSettings(generateSettings))
      : upscaleSettings
        ? addPromptHistoryItem(project.promptHistory, {
            negativePrompt: upscaleSettings.negativePromptEnabled ? upscaleSettings.negativePrompt : null,
            positivePrompt: upscaleSettings.positivePrompt,
          })
        : videoSettings
          ? addPromptHistoryItem(project.promptHistory, {
              negativePrompt: videoSettings.negativePromptEnabled ? videoSettings.negativePrompt : null,
              positivePrompt: videoSettings.positivePrompt,
            })
          : project.promptHistory,
    invocation: {
      ...project.invocation,
      destination: route.destination,
      lastSubmittedRunId: queueItemId,
      sourceId: route.sourceId,
    },
    queue: { items: [queueItem, ...project.queue.items] },
    ...(shouldResumeLiveFollowOnSubmit(project)
      ? { settings: { ...project.settings, showProgressImagesInViewer: true } }
      : {}),
    widgetGraphs:
      route.sourceId === 'generate' || route.sourceId === 'upscale' || route.sourceId === 'video'
        ? { ...project.widgetGraphs, [route.sourceId]: cloneGraph(graph) }
        : project.widgetGraphs,
  };
};

const submitInvocationSnapshot = (
  project: Project,
  backendSupportsCancellation: boolean,
  route = resolveInvocationRoute(project),
  models?: readonly ModelConfig[],
  expansion: Partial<ExpandedPositivePrompts> = {},
  workflowGenerators?: WorkflowGeneratorResolutions,
  workflowDocument?: ProjectGraphState
): Project => {
  if (!isInvocationRouteValid(route)) {
    return project;
  }

  const compiledSnapshot = compileInvocationSnapshot(project, route, models, workflowGenerators, workflowDocument);

  if (!compiledSnapshot) {
    return project;
  }

  return enqueueCompiledSnapshot(project, route, { ...compiledSnapshot, ...expansion }, backendSupportsCancellation);
};

export const createInitialWorkbenchState = (): WorkbenchState => {
  const draft = createDraftProject([]);

  return {
    account: { activeLayoutPresetId: defaultLayoutPreset.id },
    activeProjectId: draft.id,
    autosave: { status: 'idle' },
    backendConnection: { status: 'connecting' },
    notifications: [],
    projects: [draft],
    widgetFailures: [],
  };
};

export const __workbenchReducerInternal = (
  state: WorkbenchState,
  action: WorkbenchReducerAction,
  context: WorkbenchReducerContext
): WorkbenchState => {
  switch (action.type) {
    case 'createProject': {
      const project = createDraftProject(state.projects, state.account);

      return { ...state, activeProjectId: project.id, projects: [...state.projects, project] };
    }
    case 'openProject': {
      if (state.projects.some((project) => project.id === action.project.id)) {
        return { ...state, activeProjectId: action.project.id };
      }

      const project = withoutRedundantWorkingCopies(state.account, normalizeWorkbenchProject(action.project));

      return { ...state, activeProjectId: project.id, projects: [...state.projects, project] };
    }
    case 'renameProject': {
      const name = action.name.trim();

      if (!name) {
        return state;
      }

      return {
        ...state,
        projects: state.projects.map((project) => (project.id === action.projectId ? { ...project, name } : project)),
      };
    }
    case 'closeProject': {
      if (state.projects.length === 1) {
        const message = 'At least one project must remain open.';

        return addNotification(state, createNotification({ kind: 'error', message, title: 'Project close blocked' }));
      }

      const projectIndex = state.projects.findIndex((project) => project.id === action.projectId);
      const projects = state.projects.filter((project) => project.id !== action.projectId);

      if (action.projectId !== state.activeProjectId) {
        return { ...state, projects };
      }

      const fallbackProject = projects[Math.max(0, projectIndex - 1)];

      return { ...state, activeProjectId: fallbackProject.id, projects };
    }
    case 'switchProject': {
      return { ...state, activeProjectId: action.projectId };
    }
    case 'setCenterView': {
      const widgetId = getCenterWidgetIdFromViewId(action.centerViewId);

      return updateActiveWidgetRegion(state, 'center', (region) => ({
        ...region,
        activeInstanceId: region.instanceIds.includes(widgetId) ? widgetId : region.activeInstanceId,
        isCollapsed: false,
      }));
    }
    case 'applyPreset': {
      const preset = getAvailableLayoutPreset(state, action.presetId);
      const activeProject = state.projects.find((project) => project.id === state.activeProjectId);
      // Choosing the preset already in use reapplies its saved arrangement, which is how its shortcut reverts.
      const nextState =
        activeProject && activeProject.layout.presetId !== preset.id
          ? switchActiveProjectLayoutPreset(state, preset)
          : updateActiveProjectLayoutPreset(state, preset, { applyDefaultRoute: true });

      return {
        ...nextState,
        account: { ...state.account, activeLayoutPresetId: preset.id },
      };
    }
    case 'reorderLayoutPresets': {
      const layoutPresetOrder = reorderLayoutPresetIds(state.account, action.activeId, action.overId);

      return layoutPresetOrder ? { ...state, account: { ...state.account, layoutPresetOrder } } : state;
    }
    case 'addLayoutPreset': {
      const activeProject = state.projects.find((project) => project.id === state.activeProjectId);
      const presetId = action.presetId.trim();

      if (!activeProject || !presetId || isBuiltInLayoutPresetId(resolveLayoutPresetId(presetId))) {
        return state;
      }

      const preset: LayoutPreset = {
        ...(action.defaultRoute === null
          ? {}
          : {
              defaultRoute: action.defaultRoute
                ? { ...action.defaultRoute }
                : {
                    destination: activeProject.invocation.destination,
                    sourceId: activeProject.invocation.sourceId,
                  },
            }),
        iconId: action.iconId,
        id: presetId,
        label: action.label.trim() || 'Custom layout',
        snapshot: withSnapshotPresetId(createLayoutPresetSnapshot(normalizeWorkbenchProject(activeProject)), presetId),
      };
      const customLayoutPresets = [
        ...(state.account.customLayoutPresets ?? []).filter((candidate) => candidate.id !== presetId),
        preset,
      ];
      const layoutPresetOrder = [
        ...getOrderedLayoutPresets(state.account)
          .map(({ id }) => id)
          .filter((id) => id !== preset.id),
        preset.id,
      ];

      // The project moves onto the new preset, whose saved arrangement is the live one, so it opens clean; the
      // preset it came from is left as saved, its changes having gone to the new preset rather than to a copy.
      return {
        ...state,
        account: { ...state.account, activeLayoutPresetId: preset.id, customLayoutPresets, layoutPresetOrder },
        projects: state.projects.map((project) =>
          project === activeProject ? { ...project, layout: { ...project.layout, presetId: preset.id } } : project
        ),
      };
    }
    case 'saveLayoutPreset': {
      const activeProject = state.projects.find((project) => project.id === state.activeProjectId);

      if (!activeProject) {
        return state;
      }

      // The active preset saves the live arrangement; another preset saves this project's working copy of it, and
      // with no copy it has nothing unsaved to write. That copy stays (equal to the saved preset, it no longer reads
      // as unsaved): saving writes only the account, so if the write is lost the copy still holds the arrangement and
      // shows as unsaved again. Hydration drops it once the account it loads proves it redundant.
      const workingArrangement =
        activeProject.layout.presetId === action.presetId
          ? undefined
          : findLayoutPresetWorkingCopy(activeProject.presetWorkingLayouts, action.presetId);

      if (activeProject.layout.presetId !== action.presetId && !workingArrangement) {
        return state;
      }

      const snapshot = workingArrangement ?? createLayoutPresetSnapshot(normalizeWorkbenchProject(activeProject));
      const previouslySaved = resolveSavedLayoutPreset(state.account, action.presetId).snapshot;
      // Another open project's copy that equals the old saved arrangement was never a change of its own (it would now
      // show the outdated arrangement as unsaved, and saving it would overwrite this save); one equal to the new
      // arrangement is redundant. Both go. A copy that differs from both is that project's own work and stays.
      const projects = state.projects.map((project) => {
        if (project === activeProject && workingArrangement) {
          return project;
        }

        const presetWorkingLayouts = withoutPresetWorkingLayouts(project, (presetId) => {
          if (presetId !== action.presetId) {
            return false;
          }

          const copy = findLayoutPresetWorkingCopy(project.presetWorkingLayouts, presetId)!;

          return areLayoutPresetSnapshotsEqual(copy, previouslySaved) || areLayoutPresetSnapshotsEqual(copy, snapshot);
        });

        return presetWorkingLayouts === project.presetWorkingLayouts ? project : { ...project, presetWorkingLayouts };
      });

      // Built-in preset bodies are code, so their saved form lives in an
      // override map; custom presets own their snapshot outright.
      if (isBuiltInLayoutPresetId(action.presetId)) {
        const layoutPresetOverrides: LayoutPresetOverrides = {
          ...state.account.layoutPresetOverrides,
          [action.presetId]: { ...snapshot, layout: { ...snapshot.layout, presetId: action.presetId } },
        };

        return { ...state, account: { ...state.account, layoutPresetOverrides }, projects };
      }

      const customLayoutPresets = (state.account.customLayoutPresets ?? []).map((preset) =>
        preset.id === action.presetId
          ? { ...preset, snapshot: { ...snapshot, layout: { ...snapshot.layout, presetId: preset.id } } }
          : preset
      );

      return { ...state, account: { ...state.account, customLayoutPresets }, projects };
    }
    case 'restoreLayoutPresetDefault': {
      if (!isBuiltInLayoutPresetId(action.presetId)) {
        return state;
      }

      const { [action.presetId]: removedMetadata, ...layoutPresetMetadataOverrides } =
        state.account.layoutPresetMetadataOverrides ?? {};
      const { [action.presetId]: removed, ...layoutPresetOverrides } = state.account.layoutPresetOverrides ?? {};
      const { [action.presetId]: removedRoute, ...layoutPresetRouteOverrides } =
        state.account.layoutPresetRouteOverrides ?? {};

      return removedMetadata || removed || removedRoute
        ? {
            ...state,
            account: {
              ...state.account,
              layoutPresetMetadataOverrides,
              layoutPresetOverrides,
              layoutPresetRouteOverrides,
            },
          }
        : state;
    }
    case 'setLayoutPresetIcon': {
      if (isBuiltInLayoutPresetId(action.presetId)) {
        const preset = resolveSavedLayoutPreset(state.account, action.presetId);

        return setBuiltInLayoutPresetMetadata(state, action.presetId, {
          iconId: action.iconId,
          label: preset.label,
        });
      }

      return {
        ...state,
        account: {
          ...state.account,
          customLayoutPresets: (state.account.customLayoutPresets ?? []).map((preset) =>
            preset.id === action.presetId ? { ...preset, iconId: action.iconId } : preset
          ),
        },
      };
    }
    case 'setLayoutPresetRoute': {
      if (isBuiltInLayoutPresetId(action.presetId)) {
        const shippedRoute = getLayoutPreset(action.presetId).defaultRoute;
        const matchesShippedRoute =
          action.defaultRoute !== null &&
          shippedRoute !== undefined &&
          action.defaultRoute.destination === shippedRoute.destination &&
          action.defaultRoute.sourceId === shippedRoute.sourceId;

        if (action.defaultRoute === null || matchesShippedRoute) {
          const { [action.presetId]: _removed, ...layoutPresetRouteOverrides } =
            state.account.layoutPresetRouteOverrides ?? {};

          return { ...state, account: { ...state.account, layoutPresetRouteOverrides } };
        }

        return {
          ...state,
          account: {
            ...state.account,
            layoutPresetRouteOverrides: {
              ...state.account.layoutPresetRouteOverrides,
              [action.presetId]: { ...action.defaultRoute },
            },
          },
        };
      }

      return {
        ...state,
        account: {
          ...state.account,
          customLayoutPresets: (state.account.customLayoutPresets ?? []).map((preset) => {
            if (preset.id !== action.presetId) {
              return preset;
            }
            if (action.defaultRoute) {
              return { ...preset, defaultRoute: { ...action.defaultRoute } };
            }

            const { defaultRoute: _removed, ...withoutRoute } = preset;

            return withoutRoute;
          }),
        },
      };
    }
    case 'renameLayoutPreset': {
      const label = action.label.trim();

      if (!label) {
        return state;
      }

      if (isBuiltInLayoutPresetId(action.presetId)) {
        const preset = resolveSavedLayoutPreset(state.account, action.presetId);

        return setBuiltInLayoutPresetMetadata(state, action.presetId, {
          iconId: preset.iconId ?? '',
          label,
        });
      }

      return {
        ...state,
        account: {
          ...state.account,
          customLayoutPresets: (state.account.customLayoutPresets ?? []).map((preset) =>
            preset.id === action.presetId ? { ...preset, label } : preset
          ),
        },
      };
    }
    case 'deleteLayoutPreset': {
      const layoutPresetOrder = getOrderedLayoutPresets(state.account)
        .map(({ id }) => id)
        .filter((id) => id !== action.presetId);
      const customLayoutPresets = (state.account.customLayoutPresets ?? []).filter(
        (preset) => preset.id !== action.presetId
      );
      const defaultPreset = resolveSavedLayoutPreset(state.account, defaultLayoutPreset.id);
      // The deleted preset's arrangements go with it. A project that was on it moves to the default preset as a switch
      // would: its working copy of the default if it has one, otherwise the saved default.
      const projects = state.projects.map((project) => {
        if (project.layout.presetId !== action.presetId) {
          const presetWorkingLayouts = withoutPresetWorkingLayouts(project, (presetId) => presetId === action.presetId);

          return presetWorkingLayouts === project.presetWorkingLayouts ? project : { ...project, presetWorkingLayouts };
        }

        const defaultCopy = findLayoutPresetWorkingCopy(project.presetWorkingLayouts, defaultPreset.id);

        return {
          ...applyLayoutPresetToProject(
            project,
            defaultCopy ? { ...defaultPreset, snapshot: defaultCopy } : defaultPreset
          ),
          presetWorkingLayouts: withoutPresetWorkingLayouts(
            project,
            (presetId) => presetId === action.presetId || presetId === defaultPreset.id
          ),
        };
      });

      return {
        ...state,
        account: {
          ...state.account,
          activeLayoutPresetId:
            state.account.activeLayoutPresetId === action.presetId
              ? defaultLayoutPreset.id
              : state.account.activeLayoutPresetId,
          customLayoutPresets,
          layoutPresetOrder,
        },
        projects,
      };
    }
    case 'resetActiveLayout': {
      return resetActiveLayoutToSaved(state);
    }
    case 'revertLayoutPreset': {
      const activeProject = state.projects.find((project) => project.id === state.activeProjectId);

      if (!activeProject) {
        return state;
      }
      if (activeProject.layout.presetId === action.presetId) {
        return resetActiveLayoutToSaved(state);
      }
      if (!findLayoutPresetWorkingCopy(activeProject.presetWorkingLayouts, action.presetId)) {
        return state;
      }

      return updateActiveProject(state, (project) => ({
        ...pushUndo(project, 'Update layout'),
        presetWorkingLayouts: withoutPresetWorkingLayouts(project, (presetId) => presetId === action.presetId),
      }));
    }
    case 'recoverShellLayout': {
      return updateActiveLayout(state, (layout) => ({
        ...layout,
        panels: { isLeftOpen: true, isRightOpen: true, isBottomOpen: true },
      }));
    }
    case 'setInvocationSource': {
      if (!isInvocationSourceAvailable(action.sourceId)) {
        return state;
      }

      return updateActiveInvocation(state, (invocation) => ({ ...invocation, sourceId: action.sourceId }));
    }
    case 'setInvocationDestination': {
      return updateActiveInvocation(state, (invocation) => ({ ...invocation, destination: action.destination }));
    }
    case 'toggleRoutingLock': {
      return updateActiveInvocation(state, (invocation) => {
        const isLocked = invocation.sourceLocked || invocation.destinationLocked;

        return { ...invocation, destinationLocked: !isLocked, sourceLocked: !isLocked };
      });
    }
    case 'toggleSourceLock': {
      return updateActiveInvocation(state, (invocation) => ({ ...invocation, sourceLocked: !invocation.sourceLocked }));
    }
    case 'toggleDestinationLock': {
      return updateActiveInvocation(state, (invocation) => ({
        ...invocation,
        destinationLocked: !invocation.destinationLocked,
      }));
    }
    case 'openRegionWidget': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) => {
        const region = project.widgetRegions[action.region];
        const existingInstanceInRegion = region.instanceIds
          .map((instanceId) => project.widgetInstances[instanceId])
          .find((instance) => instance?.typeId === action.widgetId);
        const existingInstance =
          existingInstanceInRegion ??
          Object.values(project.widgetInstances).find((instance) => instance.typeId === action.widgetId);
        const instanceId =
          action.createNew || !existingInstance ? createId(`widget-${action.widgetId}`) : existingInstance.id;
        const widgetInstances = project.widgetInstances[instanceId]
          ? project.widgetInstances
          : {
              ...project.widgetInstances,
              [instanceId]: createWidgetInstance(action.widgetId, instanceId, action.initialValues),
            };
        // An explicit open docks a floating instance where it came from before placing it here, so it is never
        // both a window and a region member.
        const docked = restoreFloatingInstance({ ...project, widgetInstances }, instanceId, { reveal: false });
        const placed = docked.widgetRegions[action.region].instanceIds.includes(instanceId)
          ? docked
          : updateRegionOrder(docked, action.region, (slots) => [...slots, { instanceId, isFloating: false }]);
        const widgetRegions = {
          ...placed.widgetRegions,
          [action.region]: { ...placed.widgetRegions[action.region], activeInstanceId: instanceId, isCollapsed: false },
        };

        // One reveal, for the placement that ends up in front.
        return applyAutoRouteForWidgetReveal(
          {
            ...placed,
            layout: openPanelForRegion(placed.layout, action.region),
            widgetRegions: action.region === 'center' ? yieldSidePanelsTo(widgetRegions, instanceId) : widgetRegions,
          },
          action.widgetId,
          context
        );
      });
    }
    case 'selectRegionWidget': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) => {
        const region = project.widgetRegions[action.region];

        // A request that names an instance the region no longer holds — it floated, or left — is stale. It changes
        // nothing; in particular it must not discard the window.
        if (!region.instanceIds.includes(action.widgetId)) {
          return project;
        }

        if (action.region === 'center') {
          return applyAutoRouteForRevealedInstance(
            {
              ...project,
              widgetRegions: {
                ...project.widgetRegions,
                center: { ...region, activeInstanceId: action.widgetId, isCollapsed: false },
              },
            },
            action.widgetId,
            context
          );
        }

        // Expanding the active tab reveals its widget and may route; collapsing reveals nothing.
        if (region.activeInstanceId === action.widgetId) {
          const disclosed = {
            ...project,
            layout: openPanelForRegion(project.layout, action.region),
            widgetRegions: {
              ...project.widgetRegions,
              [action.region]: { ...region, isCollapsed: !region.isCollapsed },
            },
          };

          return region.isCollapsed
            ? applyAutoRouteForRevealedInstance(disclosed, action.widgetId, context)
            : disclosed;
        }

        return applyAutoRouteForRevealedInstance(
          {
            ...project,
            layout: openPanelForRegion(project.layout, action.region),
            widgetRegions: {
              ...project.widgetRegions,
              [action.region]: { ...region, activeInstanceId: action.widgetId, isCollapsed: false },
            },
          },
          action.widgetId,
          context
        );
      });
    }
    case 'toggleRegionWidget': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) => {
        const previousRegion = project.widgetRegions[action.region];
        const isEnabled = previousRegion.instanceIds.includes(action.widgetId);

        // A refused toggle must stay a no-op down to object identity: the persistence layer treats any new
        // `projects` reference as a change worth autosaving. The work surface keeps its last view, and a floating
        // instance belongs to no region.
        if (
          isEnabled
            ? action.region === 'center' && previousRegion.instanceIds.length === 1
            : project.floatingWidgets?.[action.widgetId] !== undefined
        ) {
          return project;
        }

        const nextProject = updateProjectWidgetRegion(
          updateRegionOrder(project, action.region, (slots) =>
            isEnabled
              ? slots.filter((slot) => slot.instanceId !== action.widgetId)
              : [...slots, { instanceId: action.widgetId, isFloating: false }]
          ),
          action.region,
          (region) => ({
            ...region,
            activeInstanceId: isEnabled
              ? (getNextInstanceId(previousRegion, action.widgetId) ??
                emptiedActiveInstanceId(action.region, previousRegion))
              : action.widgetId,
            isCollapsed:
              action.region === 'center' ? false : region.instanceIds.length === 0 ? true : region.isCollapsed,
          })
        );

        return applyAutoRouteForRegionFront(nextProject, previousRegion, action.region, context);
      });
    }
    case 'floatWidget': {
      return updateActiveProject(state, (project) => {
        if (project.floatingWidgets?.[action.instanceId] || !project.widgetInstances[action.instanceId]) {
          return project;
        }

        const memberRegions = WIDGET_REGIONS.filter((regionId) =>
          project.widgetRegions[regionId].instanceIds.includes(action.instanceId)
        );
        // The window returns to the chrome the float was asked from; without that hint, to its first member region.
        // The reducer accepts a center origin; the UI float control only offers dockable panel origins.
        const returnRegion = action.region && memberRegions.includes(action.region) ? action.region : memberRegions[0];

        if (!returnRegion) {
          return project;
        }

        // The window reopens where it last floated, brought on screen if that was in a larger viewport; a first
        // float takes the next cascade slot.
        const remembered = project.lastFloatingGeometry?.[action.instanceId];
        let floatingWidgets: Record<WidgetInstanceId, FloatingWidgetState> | undefined = {
          ...project.floatingWidgets,
          [action.instanceId]: toFloatingWidgetState({
            ...(remembered
              ? action.viewport
                ? fitWindowIntoViewport(remembered, action.viewport)
                : remembered
              : cascadeDefaultGeometry(Object.keys(project.floatingWidgets ?? {}).length)),
            mode: 'windowed',
            returnRegion,
            stackOrder: nextStackOrder(project.floatingWidgets),
          }),
        };
        const widgetRegions = { ...project.widgetRegions };

        // A floating instance belongs to no region: it leaves every one that held it, and only the return region
        // keeps its slot, as a marker.
        for (const regionId of memberRegions) {
          const region = project.widgetRegions[regionId];
          const slots = getRegionOrder(regionId, region.instanceIds, project.floatingWidgets);
          const order = writeRegionOrder(
            regionId === returnRegion
              ? slots.map((slot) => (slot.instanceId === action.instanceId ? { ...slot, isFloating: true } : slot))
              : slots.filter((slot) => slot.instanceId !== action.instanceId),
            floatingWidgets
          );

          floatingWidgets = order.floatingWidgets;
          widgetRegions[regionId] = withoutFloatedInstances(regionId, region, order.instanceIds);
        }

        return applyAutoRouteForRevealedInstance(
          {
            ...project,
            floatingWidgets,
            // The memory is spent while the window is open: its geometry is the window's own state again.
            lastFloatingGeometry: rememberClosedWindows(
              project.lastFloatingGeometry,
              project.floatingWidgets,
              floatingWidgets
            ),
            widgetRegions,
          },
          // The window lands on top of everything, so it is the revealed
          // surface — not the tab the rail promotes behind it.
          action.instanceId,
          context
        );
      });
    }
    case 'dockFloatingWidget': {
      return updateActiveProject(state, (project) =>
        project.floatingWidgets?.[action.instanceId]
          ? applyAutoRouteForRevealedInstance(
              restoreFloatingInstance(project, action.instanceId, { reveal: true }),
              action.instanceId,
              context
            )
          : project
      );
    }
    case 'closeFloatingWidget': {
      return updateActiveProject(state, (project) => {
        const floating = project.floatingWidgets?.[action.instanceId];

        if (!floating) {
          return project;
        }

        // Removing closes the window and frees its rail slot. It opens no panel, but a center left empty by this
        // instance gets its view back rather than staying blank.
        const closed = updateRegionOrder(project, floating.returnRegion, (slots) =>
          slots.filter((slot) => slot.instanceId !== action.instanceId)
        );

        return isAwaitedCenterView(project.widgetRegions.center, action.instanceId)
          ? updateRegionOrder(closed, 'center', (slots) => [
              ...slots,
              { instanceId: action.instanceId, isFloating: false },
            ])
          : closed;
      });
    }
    case 'setFloatingWidgetGeometry': {
      return updateActiveProject(state, (project) => {
        const floating = project.floatingWidgets?.[action.instanceId];

        if (!floating) {
          return project;
        }

        const geometry = clampSizeToMinimum({
          heightPx: action.heightPx,
          widthPx: action.widthPx,
          x: action.x,
          y: action.y,
        });

        // Ignore unchanged geometry so clicking window chrome does not dirty or autosave the project.
        if (
          floating.heightPx === geometry.heightPx &&
          floating.widthPx === geometry.widthPx &&
          floating.x === geometry.x &&
          floating.y === geometry.y
        ) {
          return project;
        }

        return {
          ...project,
          floatingWidgets: { ...project.floatingWidgets, [action.instanceId]: { ...floating, ...geometry } },
        };
      });
    }
    case 'setFloatingWidgetMode': {
      return updateActiveProject(state, (project) => {
        const floating = project.floatingWidgets?.[action.instanceId];

        if (!floating || floating.mode === action.mode) {
          return project;
        }

        return {
          ...project,
          floatingWidgets: { ...project.floatingWidgets, [action.instanceId]: { ...floating, mode: action.mode } },
        };
      });
    }
    case 'raiseFloatingWidget': {
      return updateActiveProject(state, (project) => {
        const floatingWidgets = raiseFloatingWidget(project.floatingWidgets, action.instanceId);

        // Every press and every focus arriving inside the window asks for this; only an actual raise should
        // re-route or dirty the project.
        return floatingWidgets === project.floatingWidgets
          ? project
          : applyAutoRouteForRevealedInstance({ ...project, floatingWidgets }, action.instanceId, context);
      });
    }
    case 'revealFloatingWidget': {
      return updateActiveProject(state, (project) => {
        const raised = raiseFloatingWidget(project.floatingWidgets, action.instanceId);
        const floating = raised?.[action.instanceId];
        // A shaded window is rolled up to its title bar; raising it alone would still hide its content.
        const floatingWidgets: Project['floatingWidgets'] =
          floating?.mode === 'shaded' ? { ...raised, [action.instanceId]: { ...floating, mode: 'windowed' } } : raised;

        return floatingWidgets === project.floatingWidgets
          ? project
          : applyAutoRouteForRevealedInstance({ ...project, floatingWidgets }, action.instanceId, context);
      });
    }
    case 'moveWidgetInstance': {
      return updateActiveProject(state, (project) => {
        const fromRegion = project.widgetRegions[action.fromRegion];

        // Only a docked member can be dragged; a stale request for one that has left or floated changes nothing.
        if (!fromRegion.instanceIds.includes(action.instanceId)) {
          return project;
        }

        const detached = updateRegionOrder(project, action.fromRegion, (slots) =>
          slots.filter((slot) => slot.instanceId !== action.instanceId)
        );
        // `toIndex` counts docked members; the instance lands in front of the member at that index, wherever the
        // region's markers sit.
        const moved = updateRegionOrder(detached, action.toRegion, (slots) => {
          const others = slots.filter((slot) => slot.instanceId !== action.instanceId);
          const target = others.filter((slot) => !slot.isFloating)[Math.max(0, action.toIndex)];
          const position = target ? others.indexOf(target) : others.length;

          return [
            ...others.slice(0, position),
            { instanceId: action.instanceId, isFloating: false },
            ...others.slice(position),
          ];
        });
        const nextFromInstanceIds = detached.widgetRegions[action.fromRegion].instanceIds;

        return applyAutoRouteForRevealedInstance(
          {
            ...moved,
            layout: openPanelForRegion(project.layout, action.toRegion),
            widgetRegions: {
              ...moved.widgetRegions,
              [action.fromRegion]: {
                ...fromRegion,
                activeInstanceId:
                  fromRegion.activeInstanceId === action.instanceId
                    ? (nextFromInstanceIds[0] ?? emptiedActiveInstanceId(action.fromRegion, fromRegion))
                    : fromRegion.activeInstanceId,
                instanceIds: nextFromInstanceIds,
                isCollapsed:
                  action.fromRegion === 'center' ? false : nextFromInstanceIds.length === 0 || fromRegion.isCollapsed,
              },
              [action.toRegion]: {
                ...project.widgetRegions[action.toRegion],
                activeInstanceId: action.instanceId,
                instanceIds: moved.widgetRegions[action.toRegion].instanceIds,
                isCollapsed: false,
              },
            },
          },
          // The dragged widget is revealed in the target; the promoted source tab stays behind it.
          action.instanceId,
          context
        );
      });
    }
    case 'reorderWidgetInstances': {
      return updateActiveProject(state, (project) => {
        const previousRegion = project.widgetRegions[action.region];
        const members = new Set(previousRegion.instanceIds);

        // A reorder only permutes the docked members, which then fill the docked positions around stationary
        // markers. Anything else is stale: it could re-add an instance that floated or left.
        if (
          action.instanceIds.length !== members.size ||
          new Set(action.instanceIds).size !== members.size ||
          !action.instanceIds.every((instanceId) => members.has(instanceId))
        ) {
          return project;
        }

        const nextProject = updateProjectWidgetRegion(project, action.region, (region) => ({
          ...region,
          activeInstanceId: action.activeInstanceId ?? region.activeInstanceId,
          instanceIds: action.instanceIds,
        }));

        // A reorder that also changes which tab is selected reveals that tab;
        // a pure reorder leaves the same panel in front and must not re-route.
        return applyAutoRouteForRegionFront(nextProject, previousRegion, action.region, context);
      });
    }
    case 'setWidgetInstanceAlignment': {
      return updateActiveProject(state, (project) =>
        updateProjectWidgetRegion(project, action.region, (region) => {
          const current = region.alignEndInstanceIds ?? [];
          const isAlignedEnd = current.includes(action.instanceId);

          if (action.align === 'end' ? isAlignedEnd : !isAlignedEnd) {
            return region;
          }

          const next =
            action.align === 'end'
              ? [...current, action.instanceId]
              : current.filter((instanceId) => instanceId !== action.instanceId);

          return { ...region, alignEndInstanceIds: next };
        })
      );
    }
    case 'setRegionWidgetCollapsed': {
      if (action.region === 'center') {
        return state;
      }

      return updateActiveWidgetRegion(state, action.region, (region) =>
        region.isCollapsed === action.isCollapsed ? region : { ...region, isCollapsed: action.isCollapsed }
      );
    }
    case 'setRegionWidgetSize': {
      const sizePx = clampPanelSize(action.region, action.sizePx);

      return updateActiveWidgetRegion(state, action.region, (region) =>
        region.sizePx === sizePx ? region : { ...region, sizePx }
      );
    }
    case 'setGenerateSettings': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) => {
        const updated = updateProjectWidgetValues(project, 'generate', () => cloneGenerateWidgetValues(action.values));

        // Whole-values commits come from model selection/recall — always intent-bearing.
        return updated === project || action.origin === 'system'
          ? updated
          : applyAutoRouteForGenerateEdit(updated, context);
      });
    }
    case 'patchGenerateSettings': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) => {
        const updated = updateProjectWidgetValues(project, 'generate', (values) => patchRecord(values, action.values));

        if (updated === project || action.origin === 'system') {
          return updated;
        }

        const changedKeys = getChangedValueKeys(getProjectWidgetValues(project, 'generate'), action.values);

        return isHighConfidenceGenerateEdit(changedKeys) ? applyAutoRouteForGenerateEdit(updated, context) : updated;
      });
    }
    case 'patchProjectPromptDraft': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) => {
        const updated = updateProjectWidgetValues(project, 'generate', (values) =>
          applyProjectPromptDraft(values, action.values)
        );

        if (updated === project || action.origin === 'system') {
          return updated;
        }

        return action.sourceId === 'generate'
          ? applyAutoRouteForGenerateEdit(updated, context)
          : applyAutoRouteForEdit(updated, action.sourceId, context);
      });
    }
    case 'setGenerateBatchCount': {
      const batchCount = sanitizeBatchCount(action.batchCount);

      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) =>
        updateProjectWidgetValues(project, 'generate', (values) =>
          sanitizeBatchCount(values.batchCount) === batchCount ? values : { ...values, batchCount }
        )
      );
    }
    case 'addPromptToHistory': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) => ({
        ...project,
        promptHistory: addPromptHistoryItem(project.promptHistory, action.prompt),
      }));
    }
    case 'removePromptFromHistory': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) => ({
        ...project,
        promptHistory: removePromptHistoryItem(project.promptHistory, action.prompt),
      }));
    }
    case 'clearPromptHistory': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) => ({
        ...project,
        promptHistory: [],
      }));
    }
    case 'patchWidgetValues': {
      // Generic widget-owned UI state (panel modes, tabs, sizes). Not undoable.
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) => {
        const updated = updateProjectWidgetValues(project, action.widgetId, (values) =>
          patchRecord(values, action.values)
        );

        if (updated === project || action.origin === 'system') {
          return updated;
        }

        const changedKeys = getChangedValueKeys(getProjectWidgetValues(project, action.widgetId), action.values);

        if (action.widgetId === 'generate' && isHighConfidenceGenerateEdit(changedKeys)) {
          return applyAutoRouteForGenerateEdit(updated, context);
        }
        if (action.widgetId === 'upscale' && isHighConfidenceUpscaleEdit(changedKeys)) {
          return applyAutoRouteForEdit(updated, 'upscale', context);
        }
        if (action.widgetId === 'video' && isHighConfidenceVideoEdit(changedKeys)) {
          return applyAutoRouteForEdit(updated, 'video', context);
        }

        return updated;
      });
    }
    case 'patchWidgetInstanceValues': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) =>
        updateProjectWidgetInstanceValues(project, action.instanceId, (values) =>
          patchRecord(values, cloneRecord(action.values))
        )
      );
    }
    case 'setWidgetInstanceValues': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) =>
        updateProjectWidgetInstanceValues(project, action.instanceId, (currentValues) => {
          const values = cloneRecord(action.values);

          return areRecordsShallowEqual(currentValues, values) ? currentValues : values;
        })
      );
    }
    case 'applyWorkflowAction': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) => {
        const workflowId = action.workflowId ?? project.workflows.activeWorkflowId;
        const result = applyProjectWorkflowAction(project, workflowId, action.action, now());

        if (!result.didChange) {
          return project;
        }

        // Editing an inactive workflow (a fenced async completion) must not re-route the project to it.
        return workflowId === project.workflows.activeWorkflowId && isHighConfidenceGraphEdit(action.action)
          ? applyAutoRouteForEdit(result.project, 'workflow', context)
          : result.project;
      });
    }
    case 'addProjectWorkflow': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) => {
        if (findProjectWorkflow(project, action.document.id)) {
          return project;
        }

        const nextProject = addProjectWorkflow(applyAutoRouteForEdit(project, 'workflow', context), action.document, {
          reusePlaceholder: action.reusePlaceholder,
          source: action.source,
        });

        return {
          ...nextProject,
          events: prependProjectEvent(nextProject.events, {
            createdAt: now(),
            id: createId('event'),
            summary: `Added workflow "${action.document.name || 'Untitled Workflow'}" (${action.label})`,
            type: 'graph-replaced',
          }),
        };
      });
    }
    case 'selectProjectWorkflow': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) =>
        selectProjectWorkflow(project, action.workflowId)
      );
    }
    case 'duplicateProjectWorkflow': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) =>
        duplicateProjectWorkflow(project, action.workflowId, action.copyId, () => action.copyName)
      );
    }
    case 'removeProjectWorkflow': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) =>
        removeProjectWorkflow(project, action.workflowId)
      );
    }
    case 'replaceProjectWorkflowDocument': {
      return updateProjectById(state, action.projectId, (project) =>
        selectProjectWorkflow(
          replaceProjectWorkflowDocument(
            project,
            action.workflowId,
            action.document,
            { label: action.label, source: action.source },
            now()
          ),
          action.workflowId
        )
      );
    }
    case 'setProjectWorkflowSource': {
      return updateProjectById(state, action.projectId, (project) =>
        setProjectWorkflowSource(project, action.workflowId, action.source)
      );
    }
    case 'undoWorkflowChange': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) =>
        undoProjectWorkflow(project, action.workflowId ?? project.workflows.activeWorkflowId, now())
      );
    }
    case 'redoWorkflowChange': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) =>
        redoProjectWorkflow(project, action.workflowId ?? project.workflows.activeWorkflowId, now())
      );
    }
    case 'submitInvocationSnapshot': {
      return withEnqueueNotification(
        state,
        updateActiveProject(state, (project) =>
          submitInvocationSnapshot(project, action.backendSupportsCancellation, undefined, action.models)
        ),
        state.activeProjectId
      );
    }
    case 'submitResolvedInvocationSnapshot': {
      const target = state.projects.find((project) => project.id === action.projectId);
      const workflow =
        target && action.route.sourceId === 'workflow'
          ? action.workflowId
            ? findProjectWorkflow(target, action.workflowId)
            : getActiveProjectWorkflow(target)
          : undefined;

      // The originating workflow was removed while the submission was being prepared; nothing else may run in its
      // place.
      if (target && action.route.sourceId === 'workflow' && !workflow) {
        return addNotification(
          state,
          createNotification({
            category: 'enqueue',
            kind: 'error',
            message: 'The workflow was removed from the project before it could be queued.',
            messageKey: 'widgets.workflow.submitRemovedBody',
            projectId: target.id,
            title: 'Workflow not queued',
            titleKey: 'widgets.workflow.submitRemovedTitle',
          })
        );
      }

      return withEnqueueNotification(
        state,
        updateProjectById(state, action.projectId, (project) =>
          submitInvocationSnapshot(
            project,
            action.backendSupportsCancellation,
            resolveInvocationRoute(project, 'global', action.route, action.models, workflow?.document),
            action.models,
            { positivePrompts: action.positivePrompts, positivePromptsSampleSeed: action.positivePromptsSampleSeed },
            action.workflowGenerators,
            workflow?.document
          )
        ),
        action.projectId
      );
    }
    case 'markQueueItemBackendSubmitted': {
      return updateProjectById(state, action.projectId, (project) => {
        const previousSelectedSlot = getSelectedCanvasStagingSlot(project);
        const nextProject = updateQueueItem(project, action.queueItemId, (item) => {
          const status = item.status === 'cancelled' ? 'cancelled' : 'running';
          const hasSameBackendItemIds =
            item.backendItemIds?.length === action.backendItemIds.length &&
            item.backendItemIds.every((id, index) => id === action.backendItemIds[index]);

          return item.backendBatchId === action.backendBatchId && hasSameBackendItemIds && item.status === status
            ? item
            : { ...item, backendBatchId: action.backendBatchId, backendItemIds: action.backendItemIds, status };
        });

        return clampCanvasStagingSelection(nextProject, previousSelectedSlot);
      });
    }
    case 'setQueueItemStatus': {
      const project = state.projects.find((project) => project.id === action.projectId);
      const queueItem = project?.queue.items.find((item) => item.id === action.queueItemId);

      if (queueItem?.status === 'cancelled' && action.status !== 'cancelled') {
        return state;
      }

      if (queueItem?.status === action.status && queueItem.error === action.error) {
        return state;
      }

      const nextState = updateProjectById(state, action.projectId, (project) => {
        const previousSelectedSlot = getSelectedCanvasStagingSlot(project);
        const nextProject = updateQueueItem(project, action.queueItemId, (item) => ({
          ...item,
          error: action.error,
          status: action.status,
        }));

        return clampCanvasStagingSelection(nextProject, previousSelectedSlot);
      });

      if (action.notify === false || (action.status !== 'failed' && action.status !== 'cancelled')) {
        return nextState;
      }

      return addNotification(
        nextState,
        createNotification({
          category: 'run-outcome',
          kind: action.status === 'failed' ? 'error' : 'info',
          message: action.error ?? `Queue item ${action.queueItemId} ${action.status}.`,
          projectId: action.projectId,
          title: action.status === 'failed' ? 'Invocation failed' : 'Invocation cancelled',
        })
      );
    }
    case 'routeQueueItemPartialResults': {
      const project = state.projects.find((project) => project.id === action.projectId);
      const queueItem = project?.queue.items.find((item) => item.id === action.queueItemId);

      if (queueItem?.status === 'cancelled' || queueItem?.status === 'completed') {
        return state;
      }

      return updateProjectById(state, action.projectId, (project) =>
        routeQueueItemPartialResults(project, action.queueItemId, action.backendItemId, action.images, action.videos)
      );
    }
    case 'markQueueItemBackendCancelled': {
      const project = state.projects.find((project) => project.id === action.projectId);
      const queueItem = project?.queue.items.find((item) => item.id === action.queueItemId);

      if (queueItem?.status === 'cancelled' || queueItem?.status === 'completed') {
        return state;
      }

      return updateProjectById(state, action.projectId, (project) => {
        const previousSelectedSlot = getSelectedCanvasStagingSlot(project);
        const nextProject = updateQueueItem(project, action.queueItemId, (item) => {
          const cancelledBackendItemIds = mergeBackendItemId(item.cancelledBackendItemIds, action.backendItemId);

          return {
            ...item,
            cancelledBackendItemIds,
            status: getQueueItemStatusAfterBackendCancellation(item, cancelledBackendItemIds),
          };
        });

        return clampCanvasStagingSelection(nextProject, previousSelectedSlot);
      });
    }
    case 'setQueueItemCancellationPending': {
      return updateProjectById(state, action.projectId, (project) =>
        updateQueueItem(project, action.queueItemId, (item) =>
          item.cancellationPending === action.pending
            ? item
            : { ...item, cancellationPending: action.pending || undefined }
        )
      );
    }
    case 'setQueueItemLocalRecoveryState': {
      return updateProjectById(state, action.projectId, (project) =>
        updateQueueItem(project, action.queueItemId, (item) =>
          item.localRecoveryState === action.state || item.localRecoveryState === 'durable'
            ? item
            : { ...item, localRecoveryState: action.state }
        )
      );
    }
    case 'routeQueueItemResults': {
      const project = state.projects.find((project) => project.id === action.projectId);
      const queueItem = project?.queue.items.find((item) => item.id === action.queueItemId);

      if (queueItem?.status === 'cancelled') {
        return state;
      }

      const nextState = updateProjectById(state, action.projectId, (project) =>
        routeQueueItemResults(project, action.queueItemId, action.images, action.videos)
      );

      if (action.images.length === 0) {
        return nextState;
      }

      return addNotification(
        nextState,
        createNotification({
          kind: 'success',
          message: `${action.images.length} image(s) routed from ${action.queueItemId}.`,
          projectId: action.projectId,
          title: 'Invocation completed',
        })
      );
    }
    case 'restoreQueueItemsFromJournal': {
      return updateProjectById(state, action.projectId, (project) => {
        const restoredById = new Map<string, QueueItem>();

        for (const candidate of action.items) {
          const item = normalizeRestoredQueueItem(candidate);
          if (item && !restoredById.has(item.id)) {
            restoredById.set(item.id, { ...item, localRecoveryState: 'durable' });
          }
        }

        const liveItems = project.queue.items.map((item) =>
          restoredById.delete(item.id) && item.localRecoveryState !== 'durable'
            ? { ...item, localRecoveryState: 'durable' as const }
            : item
        );
        const restoredItems: QueueItem[] = [];

        for (const item of restoredById.values()) {
          restoredItems.push(item);
        }

        return restoredItems.length === 0 && liveItems.every((item, index) => item === project.queue.items[index])
          ? project
          : { ...project, queue: { items: [...liveItems, ...restoredItems] } };
      });
    }
    case 'appendCanvasStagingCandidate': {
      return updateProjectById(state, action.projectId, (project) =>
        appendCanvasStagingCandidate(project, action.candidate)
      );
    }
    case 'selectGalleryItem': {
      return updateGalleryValuesAndPauseLiveFollow(
        state,
        (values) => {
          const selectedImagePage =
            typeof action.selectionPage === 'number' && Number.isFinite(action.selectionPage)
              ? Math.max(0, Math.floor(action.selectionPage))
              : typeof values.galleryPage === 'number' && Number.isFinite(values.galleryPage)
                ? Math.max(0, Math.floor(values.galleryPage))
                : 0;
          const settings = getGallerySettings(values);
          const semanticKey = gallerySemanticReferenceKey(parseGallerySemanticReference(values.semanticImageQuery));
          const existingNavigationQuery =
            values.selectedImageQuery && typeof values.selectedImageQuery === 'object'
              ? (values.selectedImageQuery as Record<string, unknown>)
              : null;
          const selectedImageQuery =
            action.preserveNavigationQuery && existingNavigationQuery
              ? {
                  ...existingNavigationQuery,
                  ...(semanticKey
                    ? {
                        boardId: typeof values.selectedBoardId === 'string' ? values.selectedBoardId : 'none',
                        semanticKey,
                      }
                    : { semanticKey: null }),
                  page: selectedImagePage,
                }
              : {
                  boardId: getGallerySelectionBoardId(values),
                  galleryView: values.galleryView === 'assets' ? 'assets' : 'images',
                  imageOrderDir: settings.imageOrderDir,
                  page: selectedImagePage,
                  paginationMode: settings.paginationMode,
                  searchTerm: typeof values.searchTerm === 'string' ? values.searchTerm : '',
                  ...(semanticKey ? { semanticKey } : {}),
                  starredOnly: values.starredOnly === true,
                };
          const itemKey = toGalleryItemKey(action.item);

          return {
            ...values,
            ...(action.item.kind === 'video' ? { compareImage: null } : {}),
            selectedImage: action.item,
            selectedImageName: itemKey,
            selectedImageNames: [itemKey],
            selectedImagePage,
            selectedImageQuery,
          };
        },
        action.projectId
      );
    }
    case 'toggleGalleryItemInSelection': {
      return updateGalleryValuesAndPauseLiveFollow(
        state,
        (values) => {
          const itemKey = toGalleryItemKey(action.item);
          const selectedItemKeys = getPersistedSelectedGalleryItemKeys(values);

          if (!selectedItemKeys.includes(itemKey)) {
            const settings = getGallerySettings(values);
            const semanticKey = gallerySemanticReferenceKey(parseGallerySemanticReference(values.semanticImageQuery));
            const selectedImagePage =
              typeof values.galleryPage === 'number' && Number.isFinite(values.galleryPage)
                ? Math.max(0, Math.floor(values.galleryPage))
                : 0;

            return {
              ...values,
              ...(action.item.kind === 'video' ? { compareImage: null } : {}),
              selectedImage: action.item,
              selectedImageName: itemKey,
              selectedImageNames: [...selectedItemKeys, itemKey],
              selectedImagePage,
              selectedImageQuery: {
                boardId: getGallerySelectionBoardId(values),
                galleryView: values.galleryView === 'assets' ? 'assets' : 'images',
                imageOrderDir: settings.imageOrderDir,
                page: selectedImagePage,
                paginationMode: settings.paginationMode,
                searchTerm: typeof values.searchTerm === 'string' ? values.searchTerm : '',
                ...(semanticKey ? { semanticKey } : {}),
                starredOnly: values.starredOnly === true,
              },
            };
          }

          const remainingItemKeys = selectedItemKeys.filter((key) => key !== itemKey);
          const selectedItem = getSelectedGalleryItemFromValues(values);
          const selectedItemKey =
            typeof values.selectedImageName === 'string'
              ? canonicalizeGalleryItemKey(values.selectedImageName)
              : selectedItem
                ? toGalleryItemKey(selectedItem)
                : null;
          const wasPrimary = selectedItemKey === itemKey;

          if (!wasPrimary) {
            return {
              ...values,
              selectedImageNames: remainingItemKeys,
            };
          }

          const expectedNextPrimaryKey = remainingItemKeys[remainingItemKeys.length - 1] ?? null;
          const nextPrimaryItem =
            expectedNextPrimaryKey &&
            action.nextPrimaryItem &&
            toGalleryItemKey(action.nextPrimaryItem) === expectedNextPrimaryKey
              ? action.nextPrimaryItem
              : null;
          const nextPrimaryKey = nextPrimaryItem ? toGalleryItemKey(nextPrimaryItem) : null;

          return {
            ...values,
            ...(nextPrimaryItem?.kind === 'image' ? {} : { compareImage: null }),
            selectedImage: nextPrimaryItem,
            selectedImageName: nextPrimaryKey,
            selectedImageNames: expectedNextPrimaryKey && !nextPrimaryItem ? [] : remainingItemKeys,
          };
        },
        action.projectId
      );
    }
    case 'setGalleryMultiSelection': {
      return updateGalleryValuesAndPauseLiveFollow(
        state,
        (values) => {
          const settings = getGallerySettings(values);
          const semanticKey = gallerySemanticReferenceKey(parseGallerySemanticReference(values.semanticImageQuery));
          const hasSelectionPage = typeof action.selectionPage === 'number' && Number.isFinite(action.selectionPage);
          const selectedImagePage = hasSelectionPage
            ? Math.max(0, Math.floor(action.selectionPage as number))
            : typeof values.galleryPage === 'number' && Number.isFinite(values.galleryPage)
              ? Math.max(0, Math.floor(values.galleryPage))
              : 0;
          const existingNavigationQuery =
            values.selectedImageQuery && typeof values.selectedImageQuery === 'object'
              ? (values.selectedImageQuery as Record<string, unknown>)
              : null;

          return {
            ...values,
            ...(action.primaryItem.kind === 'video' ? { compareImage: null } : {}),
            selectedImage: action.primaryItem,
            selectedImageName: toGalleryItemKey(action.primaryItem),
            selectedImageNames: action.itemKeys,
            selectedImagePage,
            // A Gallery range captures the current listing; only a host stepping within its own window keeps the old one.
            selectedImageQuery:
              action.preserveNavigationQuery && existingNavigationQuery
                ? {
                    ...existingNavigationQuery,
                    ...(semanticKey
                      ? {
                          boardId: typeof values.selectedBoardId === 'string' ? values.selectedBoardId : 'none',
                          semanticKey,
                        }
                      : { semanticKey: null }),
                    page: selectedImagePage,
                  }
                : {
                    boardId: getGallerySelectionBoardId(values),
                    galleryView: values.galleryView === 'assets' ? 'assets' : 'images',
                    imageOrderDir: settings.imageOrderDir,
                    page: selectedImagePage,
                    paginationMode: settings.paginationMode,
                    searchTerm: typeof values.searchTerm === 'string' ? values.searchTerm : '',
                    ...(semanticKey ? { semanticKey } : {}),
                    starredOnly: values.starredOnly === true,
                  },
          };
        },
        action.projectId
      );
    }
    case 'setGalleryCompareImage': {
      const updateValues = (values: Record<string, unknown>) => ({ ...values, compareImage: action.image });

      return action.image
        ? updateGalleryValuesAndPauseLiveFollow(state, updateValues, action.projectId)
        : updateGalleryValues(state, updateValues, action.projectId);
    }
    case 'selectGalleryBoard': {
      return updateGalleryValues(
        state,
        (values) => ({
          ...values,
          galleryPage: 0,
          selectedBoardId: action.boardId,
          selectedImageNames: [],
          // A semantic search ranks within the board, so it follows the switch; a map cluster is a fixed member
          // list and ends with an actual board change.
          ...(values.selectedBoardId !== action.boardId &&
          (values.semanticImageQuery as { kind?: unknown } | null | undefined)?.kind === 'cluster'
            ? { semanticImageQuery: null, semanticSearchText: null }
            : {}),
        }),
        action.projectId
      );
    }
    case 'clearGallerySelection': {
      return updateGalleryValues(
        state,
        (values) => ({ ...values, selectedImage: null, selectedImageName: null, selectedImageNames: [] }),
        action.projectId
      );
    }
    case 'setGalleryView': {
      return updateGalleryValues(
        state,
        (values) => ({
          ...values,
          galleryPage: 0,
          galleryView: action.galleryView,
          selectedImageNames: [],
          // Actual Images/Assets switches clear ranking; an absent galleryView already means Images.
          ...((values.galleryView === 'assets' ? 'assets' : 'images') !== action.galleryView
            ? { semanticImageQuery: null, semanticSearchText: null }
            : {}),
        }),
        action.projectId
      );
    }
    case 'setGallerySearchTerm': {
      return updateGalleryValues(
        state,
        (values) => ({
          ...values,
          galleryPage: 0,
          searchTerm: action.searchTerm,
        }),
        action.projectId
      );
    }
    case 'setGalleryStarredOnly': {
      return updateGalleryValues(
        state,
        (values) => ({
          ...values,
          galleryPage: 0,
          starredOnly: action.starredOnly,
        }),
        action.projectId
      );
    }
    case 'setGallerySemanticSearchMode': {
      return updateGalleryValues(
        state,
        (values) => {
          const semanticText = typeof values.semanticSearchText === 'string' ? values.semanticSearchText : null;

          if (action.enabled === (semanticText !== null)) {
            return values;
          }

          // Retain text across mode changes; entering semantic mode applies immediately, leaving clears its
          // ranking.
          if (action.enabled) {
            const text = typeof values.searchTerm === 'string' ? values.searchTerm : '';

            return {
              ...values,
              galleryPage: 0,
              searchTerm: '',
              semanticImageQuery: toGallerySemanticTextReference(text),
              semanticSearchText: text,
            };
          }

          return {
            ...values,
            galleryPage: 0,
            searchTerm: semanticText,
            semanticImageQuery: null,
            semanticSearchText: null,
          };
        },
        action.projectId
      );
    }
    case 'setGallerySemanticSearchText': {
      return updateGalleryValues(
        state,
        (values) =>
          typeof values.semanticSearchText === 'string' ? { ...values, semanticSearchText: action.text } : values,
        action.projectId
      );
    }
    case 'commitGallerySemanticSearch': {
      return updateGalleryValues(
        state,
        (values) => {
          // Apply delayed commits only when they still match the current field and mode; a board switch keeps both.
          if (values.semanticSearchText !== action.text) {
            return values;
          }

          const reference = toGallerySemanticTextReference(action.text);

          if (
            gallerySemanticReferenceKey(reference) ===
            gallerySemanticReferenceKey(parseGallerySemanticReference(values.semanticImageQuery))
          ) {
            return values;
          }

          return { ...values, galleryPage: 0, semanticImageQuery: reference };
        },
        action.projectId
      );
    }
    case 'clearGallerySearch': {
      return updateGalleryValues(
        state,
        (values) =>
          values.searchTerm === '' &&
          (values.semanticImageQuery === null || values.semanticImageQuery === undefined) &&
          (values.semanticSearchText === null || values.semanticSearchText === undefined)
            ? values
            : { ...values, galleryPage: 0, searchTerm: '', semanticImageQuery: null, semanticSearchText: null },
        action.projectId
      );
    }
    case 'updateGallerySettings': {
      const resetsQuery = action.settings.imageOrderDir !== undefined || action.settings.paginationMode !== undefined;

      return updateGalleryValues(
        state,
        (values) => ({
          ...values,
          ...action.settings,
          ...(resetsQuery ? { galleryPage: 0 } : {}),
        }),
        action.projectId
      );
    }
    case 'setGalleryPage': {
      const galleryPage = Number.isFinite(action.page) ? Math.max(0, Math.floor(action.page)) : 0;

      return updateGalleryValues(state, (values) => ({ ...values, galleryPage }), action.projectId);
    }
    case 'setGalleryPageInfo': {
      if (!Number.isFinite(action.totalImages)) {
        return state;
      }

      return updateGalleryValues(
        state,
        (values) => {
          const totalImages = Math.max(0, action.totalImages);

          return values.galleryTotalImages === totalImages ? values : { ...values, galleryTotalImages: totalImages };
        },
        action.projectId
      );
    }
    case 'patchGalleryItems': {
      return patchGalleryItemsAcrossProjects(state, new Set(action.itemKeys), action.changes);
    }
    case 'removeGalleryItems': {
      return removeGalleryItemsFromAllProjects(state, new Set(action.itemKeys));
    }
    case 'reconcileDeletedGalleryBoard': {
      const { outcome } = action;
      const deletedItemKeys = new Set<GalleryItemKey>([
        ...outcome.deletedImageNames.map((name) => toGalleryItemKey({ kind: 'image', name })),
        ...outcome.deletedVideoNames.map((name) => toGalleryItemKey({ kind: 'video', name })),
      ]);
      const confirmedMovedItemKeys = new Set<GalleryItemKey>([
        ...outcome.deletedBoardImageNames.map((name) => toGalleryItemKey({ kind: 'image', name })),
        ...outcome.deletedBoardVideoNames.map((name) => toGalleryItemKey({ kind: 'video', name })),
        ...outcome.failedImageNames.map((name) => toGalleryItemKey({ kind: 'image', name })),
        ...outcome.failedVideoNames.map((name) => toGalleryItemKey({ kind: 'video', name })),
      ]);

      // Failed and otherwise unconfirmed local items survive. The reconciler
      // moves every locally known key not confirmed deleted to Uncategorized.
      return reconcileDeletedGalleryBoard(state, outcome.boardId, deletedItemKeys, confirmedMovedItemKeys);
    }
    case 'setGalleryProjectBoardId': {
      return updateGalleryValues(state, (values) => ({ ...values, projectBoardId: action.boardId }), action.projectId);
    }
    case 'applyCanvasProjectMutation': {
      return updateProjectById(state, action.projectId, (project) => {
        const updated = applyCanvasProjectMutation(project, action.mutation);

        return updated !== project && action.origin !== 'system' && isHighConfidenceCanvasEdit(action.mutation)
          ? applyAutoRouteForEdit(updated, 'canvas', context)
          : updated;
      });
    }
    case 'commitCanvasEdit': {
      if (!isHighConfidenceCanvasEditIntent(action.intent)) {
        return state;
      }

      return updateProjectById(state, action.projectId, (project) => applyAutoRouteForEdit(project, 'canvas', context));
    }
    case 'submitCanvasInvocationSnapshot': {
      return withEnqueueNotification(
        state,
        updateProjectById(state, action.projectId, (project) =>
          enqueueCompiledSnapshot(
            project,
            { ...project.invocation, destination: action.destination, sourceId: 'canvas' },
            {
              generate: action.generate,
              graph: action.graph,
              positivePrompts: action.positivePrompts,
              positivePromptsSampleSeed: action.positivePromptsSampleSeed,
              widgetStates: getWidgetStatesSnapshot(project.widgetInstances),
            },
            action.backendSupportsCancellation,
            action.canvas
          )
        ),
        action.projectId
      );
    }
    case 'cancelQueueItem': {
      const targetProjectId = action.projectId ?? state.activeProjectId;
      const targetProject = state.projects.find((project) => project.id === targetProjectId);
      const queueItem = targetProject?.queue.items.find((item) => item.id === action.queueItemId);
      const canCancelQueueItem = queueItem ? isCancellableQueueItem(queueItem) : false;
      const nextState = updateProjectById(state, targetProjectId, (project) => {
        const previousSelectedSlot = getSelectedCanvasStagingSlot(project);
        const nextProject: Project = {
          ...project,
          queue: {
            items: project.queue.items.map((item) => {
              if (item.id !== action.queueItemId || !isCancellableQueueItem(item)) {
                return item;
              }

              return { ...item, cancellationPending: true, status: 'cancelled' };
            }),
          },
        };

        return clampCanvasStagingSelection(nextProject, previousSelectedSlot);
      });

      if (!targetProject || !queueItem || !canCancelQueueItem) {
        return nextState;
      }

      return addNotification(
        nextState,
        createNotification({
          kind: 'info',
          message: action.notice?.message ?? `${targetProject.name}: ${action.queueItemId}`,
          projectId: targetProject.id,
          title: action.notice?.title ?? 'Invocation cancellation requested',
        })
      );
    }
    case 'cancelAllQueueItems': {
      const cancellableCount = state.projects.reduce(
        (count, project) =>
          shouldApplyQueueBulkActionToProject(project, action.projectId)
            ? count + project.queue.items.filter(isCancellableQueueItem).length
            : count,
        0
      );

      if (cancellableCount === 0) {
        return state;
      }

      const nextState: WorkbenchState = {
        ...state,
        projects: state.projects.map((project) => {
          const previousSelectedSlot = getSelectedCanvasStagingSlot(project);
          const nextProject: Project = {
            ...project,
            queue: {
              items: shouldApplyQueueBulkActionToProject(project, action.projectId)
                ? project.queue.items.map((item) =>
                    isCancellableQueueItem(item) ? { ...item, cancellationPending: true, status: 'cancelled' } : item
                  )
                : project.queue.items,
            },
          };

          return clampCanvasStagingSelection(nextProject, previousSelectedSlot);
        }),
      };

      return addNotification(
        nextState,
        createNotification({
          kind: 'info',
          message: `${cancellableCount} queue item${cancellableCount === 1 ? '' : 's'}.`,
          title: 'Invocation cancellation requested',
        })
      );
    }
    case 'cancelAllQueueItemsExceptCurrent': {
      const cancellableCount = state.projects.reduce(
        (count, project) =>
          shouldApplyQueueBulkActionToProject(project, action.projectId)
            ? count +
              project.queue.items.filter(
                (item) => isCancellableQueueItem(item) && item.id !== action.currentQueueItemId
              ).length
            : count,
        0
      );

      if (cancellableCount === 0) {
        return state;
      }

      const nextState: WorkbenchState = {
        ...state,
        projects: state.projects.map((project) => {
          const previousSelectedSlot = getSelectedCanvasStagingSlot(project);
          const nextProject: Project = {
            ...project,
            queue: {
              items: shouldApplyQueueBulkActionToProject(project, action.projectId)
                ? project.queue.items.map((item) =>
                    isCancellableQueueItem(item) && item.id !== action.currentQueueItemId
                      ? { ...item, cancellationPending: true, status: 'cancelled' }
                      : item
                  )
                : project.queue.items,
            },
          };

          return clampCanvasStagingSelection(nextProject, previousSelectedSlot);
        }),
      };

      return addNotification(
        nextState,
        createNotification({
          kind: 'info',
          message: `${cancellableCount} queue item${cancellableCount === 1 ? '' : 's'}.`,
          title: 'Invocation cancellation requested',
        })
      );
    }
    case 'clearCompletedQueueItems': {
      return {
        ...state,
        projects: state.projects.map((project) => {
          const previousSelectedSlot = getSelectedCanvasStagingSlot(project);
          const nextProject: Project = {
            ...project,
            queue: { items: project.queue.items.filter((item) => !isClearableQueueItem(item)) },
          };

          return clampCanvasStagingSelection(nextProject, previousSelectedSlot);
        }),
      };
    }
    case 'undoProjectChange': {
      return updateActiveProject(state, (project) => {
        const undoEntry = project.undoRedo.past.at(-1);

        if (!undoEntry) {
          return project;
        }

        const restoredProject = restoreUndoSnapshot(project, undoEntry.project);

        return {
          ...restoredProject,
          events: project.events,
          promptHistory: project.promptHistory,
          queue: project.queue,
          undoRedo: {
            future: [
              {
                createdAt: now(),
                id: createId('redo'),
                label: undoEntry.label,
                project: createUndoSnapshot(project),
              },
              ...project.undoRedo.future,
            ].slice(0, HISTORY_LIMIT),
            past: project.undoRedo.past.slice(0, -1),
          },
        };
      });
    }
    case 'redoProjectChange': {
      return updateActiveProject(state, (project) => {
        const redoEntry = project.undoRedo.future[0];

        if (!redoEntry) {
          return project;
        }

        const restoredProject = restoreUndoSnapshot(project, redoEntry.project);

        return {
          ...restoredProject,
          events: project.events,
          promptHistory: project.promptHistory,
          queue: project.queue,
          undoRedo: {
            future: project.undoRedo.future.slice(1),
            past: [
              ...project.undoRedo.past,
              {
                createdAt: now(),
                id: createId('undo'),
                label: redoEntry.label,
                project: createUndoSnapshot(project),
              },
            ].slice(-HISTORY_LIMIT),
          },
        };
      });
    }
    case 'hydrateWorkbench': {
      return { ...normalizeWorkbenchState(action.state), backendConnection: state.backendConnection };
    }
    case 'replaceProjectFromServer': {
      const normalizedServerProject = normalizeWorkbenchProject(action.project);
      const localProject = state.projects.find((project) => project.id === action.projectId);
      const serverProject: Project = localProject
        ? {
            ...normalizedServerProject,
            canvas: {
              ...normalizedServerProject.canvas,
              documentRevision:
                Math.max(normalizedServerProject.canvas.documentRevision, localProject.canvas.documentRevision) + 1,
            },
            queue: localProject.queue,
          }
        : normalizedServerProject;
      return {
        ...state,
        projects: localProject
          ? state.projects.map((project) => (project.id === action.projectId ? serverProject : project))
          : [...state.projects, serverProject],
      };
    }
    case 'retargetProject': {
      const liveProject = state.projects.find((candidate) => candidate.id === action.projectId);
      const project = liveProject
        ? withAuthoritativeProjectBoard(
            normalizeWorkbenchProject(
              {
                ...liveProject,
                id: action.targetProjectId,
                name: liveProject.name === action.sourceName ? action.name : liveProject.name,
              },
              { isArriving: false }
            ),
            action.boardId
          )
        : normalizeWorkbenchProject(action.project);
      const targetAlreadyOpen = state.projects.some(
        (candidate) => candidate.id === action.targetProjectId && candidate.id !== action.projectId
      );
      if (targetAlreadyOpen) {
        return state;
      }
      return {
        ...state,
        activeProjectId: state.activeProjectId === action.projectId ? project.id : state.activeProjectId,
        projects: state.projects.some((candidate) => candidate.id === action.projectId)
          ? state.projects.map((candidate) => (candidate.id === action.projectId ? project : candidate))
          : [...state.projects, project],
      };
    }
    case 'autosaveScheduled': {
      return state.autosave.status === 'pending' || state.autosave.status === 'saving'
        ? state
        : {
            ...state,
            autosave: {
              ...(state.autosave.lastSavedAt ? { lastSavedAt: state.autosave.lastSavedAt } : {}),
              status: 'pending',
            },
          };
    }
    case 'autosaveStarted': {
      return { ...state, autosave: { status: 'saving' } };
    }
    case 'autosavePending': {
      return { ...state, autosave: { error: action.error, status: 'error' } };
    }
    case 'autosaveSucceeded': {
      return { ...state, autosave: { lastSavedAt: action.savedAt, status: 'saved' } };
    }
    case 'autosaveFailed': {
      return addNotification(
        { ...state, autosave: { error: action.error, status: 'error' } },
        createNotification({ kind: 'error', message: action.error, title: 'Autosave failed' })
      );
    }
    case 'markAllNotificationsRead': {
      return {
        ...state,
        notifications: state.notifications.map((notification) => ({ ...notification, isRead: true })),
      };
    }
    case 'clearNotifications': {
      return { ...state, notifications: [] };
    }
    case 'recordWidgetFailure': {
      const hasFailure = state.widgetFailures.some((failure) => failure.widgetId === action.failure.widgetId);

      if (hasFailure) {
        return state;
      }

      return addNotification(
        {
          ...state,
          widgetFailures: [action.failure, ...state.widgetFailures],
        },
        createNotification({
          kind: 'error',
          message: action.failure.details,
          title: `Widget failed: ${action.failure.widgetId}`,
        })
      );
    }
    case 'recordError': {
      const detail = describeError(action.context?.error);
      return addNotification(
        state,
        createNotification({
          kind: 'error',
          message: detail ? `${action.message}: ${detail}` : action.message,
          title: 'Error',
        })
      );
    }
    case 'setBackendConnectionStatus': {
      const timestamp = now();

      if (state.backendConnection.status === action.status && state.backendConnection.error === action.error) {
        return state;
      }

      return {
        ...state,
        backendConnection: {
          error: action.error,
          lastConnectedAt: action.status === 'connected' ? timestamp : state.backendConnection.lastConnectedAt,
          lastDisconnectedAt: action.status === 'disconnected' ? timestamp : state.backendConnection.lastDisconnectedAt,
          status: action.status,
        },
      };
    }
    case 'recordNotice': {
      return addNotification(
        state,
        createNotification({ kind: action.kind, message: action.message, title: action.title })
      );
    }
    case 'setActiveProjectSettings': {
      return updateActiveProject(state, (project) => {
        const settings = normalizeProjectSettings({ ...project.settings, ...action.settings });
        // Explicit live-follow clears any selection-imposed pause.
        const withoutPause =
          action.settings.showProgressImagesInViewer !== undefined && getLiveFollowPausedAt(project) !== null
            ? updateProjectWidgetValues(project, 'gallery', ({ liveFollowPausedAt: _pausedAt, ...values }) => values)
            : project;

        return Object.entries(settings).every(([key, value]) => {
          const settingKey = key as keyof ProjectSettings;

          return areProjectSettingValuesEqual(
            project.settings[settingKey],
            value as ProjectSettings[typeof settingKey]
          );
        })
          ? withoutPause
          : { ...withoutPause, settings };
      });
    }
  }
};

export type __WorkbenchReducerActionInternal = WorkbenchReducerAction;
