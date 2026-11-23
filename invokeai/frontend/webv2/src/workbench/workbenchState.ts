import type { GenerateWidgetValues, ImageWithDims } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type {
  QueueCompiledSubmission,
  QueueHistoryItemStatus,
  QueueRunErrorCode,
  QueueSubmissionRetry,
} from '@features/queue/contracts';
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
  ResolvedInvocationRoute,
  ResultDestination,
  UpscaleSourceMetadata,
} from '@workbench/invocationContracts';
import type {
  FloatingWidgetMode,
  LayoutPreset,
  LayoutPresetId,
  LayoutPresetMetadataOverride,
  LayoutPresetMetadataOverrides,
  LayoutPresetOverrides,
  LayoutPresetRoute,
  LayoutPresetRouteOverrides,
  WidgetRegion,
  WidgetRegionState,
} from '@workbench/layoutContracts';
import type {
  AccountState,
  Project,
  LayoutUndoNotice,
  ProjectUndoSnapshot,
  PromptHistoryItem,
  WorkbenchNotification,
  WorkbenchNotificationAction,
  WorkbenchNotificationCategory,
  ProjectLoadResult,
  UndoRedoEntry,
  WorkbenchNotificationKind,
  WorkbenchState,
} from '@workbench/projectContracts';
import type { ProjectSettings, ScopedDefaults } from '@workbench/settings/contracts';
import type {
  WidgetFailure,
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
  gallerySemanticReferenceKey,
  getGalleryChosenAutoAddBoardId,
  getGalleryMediaFilter,
  getGallerySettings,
  toGalleryMediaKind,
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
  getGalleryItemView,
  getGalleryView,
  settleGalleryValues,
} from '@features/gallery/contracts';
import { describeError } from '@platform/logging/normalize';
import { prependProjectEvent, PROJECT_EVENT_LIMIT } from '@workbench/projectEvents';
import { createProjectId } from '@workbench/projects/ids';

import type { InvocationRuntimeInputs } from './invocationContracts';
import type { WorkbenchQueueItem as QueueItem } from './queueHistoryContracts';
import type { LayoutAvailability, ResolvedLayout } from './workspace/resolveLayout';
import type { WorkspaceStructure } from './workspace/structure';
import type { CustomizeDraft, NewWorkspaceDraft } from './workspace/structureLock';
import type { StructureAction } from './workspace/structureReducer';
import type { GuestSurface, WorkspaceView } from './workspace/view';

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
import {
  applyCanvasProjectMutation,
  pickStagedSlotAutoSwitch,
  type CanvasProjectMutation,
} from './canvasProjectMutations';
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
import { applyWidgetDefaults, getProjectWidgetValues, getResolvedProjectWidgetValues } from './widgetState';
export { nextLayerName } from './canvasProjectMutations';
import type { ProjectPromptDraftPatch } from '@features/generation/settings';

import { compileGenerateGraph, resolveGenerateSeed } from '@features/generation/graph';
import {
  addPromptHistoryItem,
  applyProjectPromptDraft,
  cloneGenerateWidgetValues,
  createDynamicPromptsSampleSeed,
  getGenerationModelAvailabilityReasons,
  getPromptDraftFromValues,
  getPromptHistoryItemFromGenerateSettings,
  migrateProjectPromptDraft,
  normalizeGenerateSettings,
  normalizeGenerateWidgetValues,
  planGenerateSubmission,
  removePromptHistoryItem,
  restorePromptHistoryItems,
  sanitizeBatchCount,
  sanitizeSampleSeed,
  syncGenerateWidgetValuesWithModels,
} from '@features/generation/settings';
import { getQueueSourceResultSelection, MAX_QUEUE_BATCH_ITEMS } from '@features/queue';
import {
  getAutoSwitchFollowedLive,
  getQueueItemSnapshotLabel,
  pickAutoSwitch,
  selectsLandedResult,
} from '@features/queue/contracts';
import {
  clearDeletedUpscaleInput,
  cloneUpscaleWidgetValues,
  compileModelOnlyUpscaleGraph,
  compileUpscaleGraph,
  createDefaultUpscaleWidgetValues,
  getSpandrelNativeScale,
  getUpscaleOutputDimensions,
  getUpscaleSubmissionSource,
  getUpscaleValidationReasons,
  isUpscaleSourceMetadataCurrent,
  normalizeUpscaleWidgetValues,
  resolveUpscaleSeed,
  resolveUpscaleSubmissionValues,
  syncUpscaleWidgetValuesWithModels,
  type SpandrelModelConfig,
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
  type CanvasStagingSlot,
  isCancellableQueueItem,
  isUnsubmittedQueueItem,
} from './canvasStagingView';
import {
  cascadeDefaultGeometry,
  fitWindowIntoViewport,
  normalizeLastFloatingGeometry,
  rememberClosedWindows,
} from './floatingWindows';
import { getNaturalDestination } from './graphWidgets';
import {
  defaultInvocationRoute,
  getDynamicPromptsExpansionSource,
  isInvocationRouteValid,
  isSameDynamicPromptsExpansionSource,
  isResultDestinationAvailable,
  resolveInvocationRoute,
} from './invocation';
import {
  getAvailableSourceIdForWidgetTypeId,
  getSourceIdForWidgetTypeId,
  isInvocationSourceAvailable,
} from './invocationSources';
import { getOrderedLayoutPresets, normalizeLayoutPresetOrder, reorderLayoutPresetIds } from './layoutPresetCollection';
import { defaultLayoutPreset, isBuiltInLayoutPresetId, layoutPresets, resolveLayoutPresetId } from './layoutPresets';
import { doesLayoutPresetExist, resolveSavedLayoutPreset } from './layoutPresetSnapshots';
import {
  BUILT_IN_PROJECT_SETTINGS,
  getEffectiveProjectSettings,
  normalizeProjectSettings,
  PROJECT_SETTINGS_OWNER,
} from './settings/projectSettings';
import {
  getWorkspaceStructure,
  isNewWorkspaceId,
  writeWorkspaceStructure,
  isSameWorkspaceRecord,
  readWorkspaceRecord,
  restoreWorkspaceDefault,
  writeWorkspaceRecord,
} from './workspace/accountWorkspaces';
import {
  beginCustomizeSession,
  editCustomizeDraft,
  finishCustomizeSession,
  getCustomizeRouteReturn,
  getRouteMove,
  hasCustomizeChanges,
  isCustomizeStep,
  LAYOUT_HISTORY_LIMIT,
  LAYOUT_MERGE_WINDOW_MS,
  pushCustomizeHistory,
  resetCustomizeSession,
  saveCustomizeSessionAsNew,
  stepCustomizeHistory,
  stepRouteMove,
} from './workspace/customizeDraft';
import { enterWorkspace } from './workspace/entry';
import {
  isRecordValue,
  loadStoredProjectLayout,
  releaseHeldLayout,
  settleArrivingProject,
  settleHydratedProjects,
  type StoredLayoutFields,
} from './workspace/legacyProjectLayout';
import {
  getProjectStructure,
  getProjectView,
  getProjectViewId,
  getProjectWorkspace,
  type LayoutSource,
} from './workspace/projectLayout';
import { getShownGuest, resolveLayout } from './workspace/resolveLayout';
import { readWrittenStructureIds } from './workspace/storedAccount';
import {
  isStoredStructure,
  readStoredOverride,
  readStoredStructure,
  withoutRetiredTypes,
} from './workspace/storedStructure';
import { getCustomizeSession, pruneCustomizeSessions } from './workspace/structureLock';
import { captureLayout, getAddedInstanceId, reduceStructure } from './workspace/structureReducer';
import {
  frontRegion,
  getGuestInstanceId,
  getGuestWindowRegion,
  getWindowView,
  isGuestSuperseded,
  normalizeViews,
  pruneView,
  raiseWindow,
  removeGuest,
  setRegionCollapsed,
  setRegionSize,
  setWindowGeometry,
  setWindowMode,
  showGuest,
  withView,
} from './workspace/view';
import { getWorkspaceJob } from './workspace/workspaceDerivations';

interface QueueGenerateSnapshot {
  negativePromptNodeId: string;
  positivePromptNodeId: string;
  seedNodeId: string;
  values: GenerateWidgetValues;
}

export interface WorkbenchReducerContext {
  autoSwitchInvocationRoute: boolean;
  /** The widget registry's answers layouts resolve with; the reducer reads the registry through nothing else. */
  availability: LayoutAvailability;
}

type WorkbenchReducerAction =
  | { type: 'createProject' }
  | { type: 'openProject'; project: Project }
  | { type: 'closeProject'; projectId: string }
  | { type: 'renameProject'; projectId: string; name: string }
  | { type: 'switchProject'; projectId: string }
  | { type: 'applyPreset'; presetId: LayoutPresetId }
  | { type: 'reorderLayoutPresets'; activeId: LayoutPresetId; overId: LayoutPresetId }
  | { type: 'restoreLayoutPresetDefault'; presetId: LayoutPresetId }
  | { type: 'deleteLayoutPreset'; presetId: LayoutPresetId }
  /**
   * Customize the active project's workspace on a draft of its structure, or with `newWorkspace` a new workspace drafted
   * from what it shows; its layout history starts afresh.
   */
  | { type: 'beginCustomize'; newWorkspace?: NewWorkspaceDraft }
  | { type: 'setCustomizeDraft'; draft: CustomizeDraft }
  /** Makes the draft a built-in's shipped structure and metadata, seen through an empty view; Done removes overrides. */
  | { type: 'resetCustomizeToDefault' }
  /** Save as new: the session's edits go on as a new workspace's draft, and the original's draft edits are dropped. */
  | { type: 'saveCustomizeAsNew'; workspace: NewWorkspaceDraft }
  /** Drops the active project's view of the workspace: every front, size, panel and window as the workspace sets them. */
  | { type: 'resetWorkspaceView'; workspaceId: LayoutPresetId }
  /** The account that adopted is saved; projects migrated with it may now be written in the new shape. */
  | { type: 'releaseHeldLayouts'; savedAccount: AccountState }
  /** Locking: no project outside Customize keeps steps that would change structure on undo. */
  | { type: 'clearLayoutHistory' }
  /** Cancel: the account and the project's view are as before Customize. */
  | { type: 'cancelCustomize' }
  /**
   * Done, in one step, from the session alone: what is on screen becomes the workspace's defaults, the draft and its
   * name, icon and route are written to the account (a new workspace is created and the project moves onto it), and
   * the project's view keeps only what those defaults cannot say. On an existing workspace it is one undoable step.
   */
  | { type: 'finishCustomize' }
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
  | { type: 'frontRegionWidget'; region: WidgetRegion; instanceId: WidgetInstanceId; projectId?: string }
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
      region?: WidgetRegion;
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
  | { type: 'revealFloatingWidget'; instanceId: WidgetInstanceId }
  /**
   * Shows a widget type the active project's workspace does not place on that surface as its guest (view only), with a
   * values record made on demand; showing it again fronts it.
   */
  | {
      type: 'showGuest';
      typeId: WidgetTypeId;
      surface: GuestSurface;
      initialValues?: Record<string, unknown>;
      viewport?: { width: number; height: number };
    }
  /** Forgets a guest; its values stay. */
  | { type: 'closeGuest'; instanceId: WidgetInstanceId }
  /** "Keep in workspace": places the guest on its surface, a guest window as a window; structural. */
  | { type: 'keepGuest'; instanceId: WidgetInstanceId; allowMultiple: boolean }
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
  | { type: 'restorePromptHistory'; items: readonly PromptHistoryItem[]; projectId: string }
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
  | {
      type: 'clearWidgetInstanceValues';
      instanceId: WidgetInstanceId;
      keys: readonly string[];
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
  | ({
      type: 'submitResolvedInvocationSnapshot';
      runtime: InvocationRuntimeInputs;
      backendSupportsCancellation: boolean;
      prepend?: boolean;
      workflowGenerators?: WorkflowGeneratorResolutions;
      route: InvocationRoute;
      models?: readonly ModelConfig[];
      /** The project the submission was prepared from; a switch in between never redirects it. */
      projectId: string;
      workflowId?: string;
      upscaleSourceMetadata?: UpscaleSourceMetadata;
    } & Partial<ExpandedPositivePrompts>)
  | {
      type: 'submitModelOnlyUpscale';
      runtime: Pick<InvocationRuntimeInputs, 'boardDestinationIds'>;
      projectId: string;
      image: ImageWithDims;
      upscaleModel: SpandrelModelConfig;
      /** `'native'` runs the upscaler once, unsized. */
      scale: number | 'native';
      sourceMetadata: Record<string, unknown> | null;
      backendSupportsCancellation: boolean;
    }
  | {
      type: 'markQueueItemBackendSubmitted';
      projectId: string;
      queueItemId: string;
      backendItemIds: number[];
      backendBatchId?: string;
    }
  | {
      type: 'recordQueueItemSubmissionRetry';
      projectId: string;
      queueItemId: string;
      retry: QueueSubmissionRetry;
    }
  | {
      type: 'setQueueItemStatus';
      projectId: string;
      queueItemId: string;
      status: QueueHistoryItemStatus;
      error?: string;
      failedBackendItemIds?: readonly number[];
      errorCode?: QueueRunErrorCode;
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
  | { type: 'noteQueueItemFirstProgressFrame'; projectId: string; queueItemId: string; itemIndex: number }
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
  | {
      type: 'reopenQueueItemForRetry';
      projectId: string;
      queueItemId: string;
      retriedItems: { itemId: number; sourceItemId: number }[];
    }
  | {
      type: 'stageRetriedCanvasResults';
      projectId: string;
      queueItemId: string;
      backendItemId: number;
      images: GeneratedImageContract[];
    }
  | { type: 'appendCanvasStagingCandidate'; projectId: string; candidate: CanvasStagingCandidateContract }
  | {
      type: 'selectGalleryItem';
      item: GalleryItem;
      /** Navigate within the item's own board, unfiltered, not the Gallery's listing. */
      navigateItemBoard?: boolean;
      preserveNavigationQuery?: boolean;
      projectId?: string;
      selectionPage?: number;
    }
  | {
      type: 'toggleGalleryItemInSelection';
      item: GalleryItem;
      nextPrimaryItem: GalleryItem | null;
      projectId?: string;
      selectionPage?: number;
    }
  | {
      type: 'setGalleryMultiSelection';
      itemKeys: GalleryItemKey[];
      primaryItem: GalleryItem;
      preserveNavigationQuery?: boolean;
      projectId?: string;
      selectionPage?: number;
    }
  | { type: 'setGalleryCompareImage'; image: GalleryImageItem | null; projectId?: string }
  | { type: 'selectGalleryBoard'; boardId: string; projectId?: string }
  | { type: 'clearGallerySelection'; projectId?: string }
  | { type: 'setGalleryView'; galleryView: 'images' | 'assets'; projectId?: string }
  | { type: 'setGallerySearchTerm'; searchTerm: string; projectId?: string }
  | { type: 'setGallerySemanticSearchMode'; enabled: boolean; projectId?: string }
  /** The semantic field's live text; the ranking follows only on commit. */
  | { type: 'setGallerySemanticSearchText'; text: string; projectId?: string }
  | { type: 'commitGallerySemanticSearch'; text: string; projectId?: string }
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
  | {
      type: 'removeGalleryItems';
      itemKeys: GalleryItemKey[];
      /** The caller selects a next image in `projectId` right after, so that project's comparison stays open. */
      keepComparisonForSuccessor?: boolean;
      projectId?: string;
    }
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
      runtime: Pick<InvocationRuntimeInputs, 'boardDestinationIds'>;
      backendSupportsCancellation: boolean;
      canvas: CanvasStateContractV3;
      destination: ResultDestination;
      generate: QueueGenerateSnapshot;
      graph: GraphContract;
      prepend?: boolean;
      projectId: string;
    } & Partial<ExpandedPositivePrompts>)
  | {
      type: 'cancelQueueItem';
      queueItemId: string;
      projectId?: string;
      /** `null` when the caller reports the stop in a notice of its own. */
      notice?: { title: string; message: string } | null;
    }
  | { type: 'cancelUnsubmittedQueueItems'; projectId?: string }
  | { type: 'clearCompletedQueueItems' }
  | {
      type: 'undoProjectChange';
      projectId?: string;
      throughEntryId?: string;
    }
  | { type: 'redoProjectChange'; projectId?: string }
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
  | { type: 'autosaveNeedsAttention'; savedAt: string }
  | { type: 'autosaveSucceeded'; savedAt: string }
  | { type: 'autosaveFailed'; error: string }
  | { type: 'markAllNotificationsRead' }
  | { type: 'markNotificationsRead'; notificationIds: readonly string[] }
  | { type: 'clearNotifications' }
  | { type: 'dismissNotification'; notificationId: string }
  | { type: 'withdrawNotificationRetry'; itemIds: readonly number[] }
  | { type: 'restoreNotifications'; notifications: readonly WorkbenchNotification[] }
  | { type: 'recordWidgetFailure'; failure: WidgetFailure }
  /** Patches the settings of `projectId` (the active project without one); a project that is not open is left alone. */
  | { type: 'setActiveProjectSettings'; settings: Partial<ProjectSettings>; projectId?: string }
  /** Mirrors the user's `scopedDefaults` preference; the settings store owns it. */
  | { type: 'setScopedDefaults'; scopedDefaults: ScopedDefaults }
  | {
      type: 'recordError';
      message: string;
      area?: string;
      context?: { error?: unknown; [key: string]: unknown };
      namespace?: LogNamespace;
      projectId?: string;
    }
  | { type: 'setBackendConnectionStatus'; status: WorkbenchState['backendConnection']['status']; error?: string }
  | {
      type: 'recordNotice';
      kind: WorkbenchNotificationKind;
      category?: WorkbenchNotificationCategory;
      title: string;
      message?: string;
      actions?: readonly WorkbenchNotificationAction[];
      persistent?: boolean;
      /** With a key, the text follows the language; `title` and `message` are its English. */
      titleKey?: string;
      titleValues?: WorkbenchNotification['titleValues'];
      messageKey?: string;
      messageValues?: WorkbenchNotification['messageValues'];
    };

const NOTIFICATION_LIMIT = 100;
const now = (): string => new Date().toISOString();

const createId = (prefix: string): string =>
  `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

const createNotification = ({
  actions,
  category,
  kind,
  message,
  messageKey,
  messageListValues,
  messageValues,
  persistent,
  projectId,
  title,
  titleKey,
  titleValues,
}: {
  actions?: readonly WorkbenchNotificationAction[];
  category?: WorkbenchNotificationCategory;
  kind: WorkbenchNotificationKind;
  message?: string;
  messageKey?: string;
  messageListValues?: WorkbenchNotification['messageListValues'];
  messageValues?: WorkbenchNotification['messageValues'];
  persistent?: boolean;
  projectId?: string;
  title: string;
  titleKey?: string;
  titleValues?: WorkbenchNotification['titleValues'];
}): WorkbenchNotification => ({
  ...(actions && actions.length > 0 ? { actions } : {}),
  ...(persistent ? { persistent } : {}),
  ...(titleValues ? { titleValues } : {}),
  ...(messageValues ? { messageValues } : {}),
  ...(messageListValues ? { messageListValues } : {}),
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

  // Reuse the newest matching error's id so repeats do not toast again; successes always toast.
  if (
    newest &&
    newest.kind === 'error' &&
    newest.kind === notification.kind &&
    newest.title === notification.title &&
    newest.message === notification.message
  ) {
    const { actions: _previousActions, persistent: _previousPersistent, ...previous } = newest;

    return {
      ...state,
      notifications: [
        {
          ...previous,
          ...(notification.actions ? { actions: notification.actions } : {}),
          ...(notification.persistent ? { persistent: notification.persistent } : {}),
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

type NotificationText = Pick<WorkbenchNotification, 'message' | 'messageKey' | 'messageValues'>;

const describeRunLabel = (queueItem: QueueItem | undefined): NotificationText =>
  withRunLabel(queueItem, (label) => ({
    message: `“${label}”`,
    messageKey: 'notifications.run.label',
    messageValues: { label },
  }));

const withRunLabel = (
  queueItem: QueueItem | undefined,
  describe: (label: string) => NotificationText,
  unlabeled: NotificationText = {}
): NotificationText => {
  const label = queueItem ? getQueueItemSnapshotLabel(queueItem) : '';

  return label ? describe(label) : unlabeled;
};

type CodedRunFailure = Exclude<QueueRunErrorCode, 'removed' | 'unexplained'>;

/** `key` for a labeled run, `${key}Unlabeled` without; `reason` is the server's or validator's text. */
const CODED_RUN_FAILURES: Record<
  CodedRunFailure,
  { key: string; labeled: (label: string, reason: string) => string; unlabeled: (reason: string) => string }
> = {
  'invalid-submission': {
    key: 'notifications.runOutcome.invalidSubmissionMessage',
    labeled: (label, reason) => `“${label}” could not be submitted because its queued request is invalid: ${reason}`,
    unlabeled: (reason) => `The run could not be submitted because its queued request is invalid: ${reason}`,
  },
  'not-accepted': {
    key: 'notifications.runOutcome.notAcceptedMessage',
    labeled: (label) => `The server queue did not accept “${label}”. The queue may be full.`,
    unlabeled: () => 'The server queue did not accept the run. The queue may be full.',
  },
  'project-not-saved': {
    key: 'notifications.runOutcome.projectNotSavedMessage',
    labeled: (label) => `“${label}” was not submitted because its project cannot be saved.`,
    unlabeled: () => 'The run was not submitted because its project cannot be saved.',
  },
  'server-error': {
    key: 'notifications.runOutcome.serverErrorMessage',
    labeled: (label, reason) => `The server kept failing to queue “${label}”: ${reason}`,
    unlabeled: (reason) => `The server kept failing to queue the run: ${reason}`,
  },
};

const describeRunFailure = (
  queueItem: QueueItem | undefined,
  reason: string,
  errorCode: QueueRunErrorCode | undefined
): NotificationText => {
  if (errorCode && errorCode !== 'removed' && errorCode !== 'unexplained') {
    const { key, labeled, unlabeled } = CODED_RUN_FAILURES[errorCode];
    return withRunLabel(
      queueItem,
      (label) => ({ message: labeled(label, reason), messageKey: key, messageValues: { label, reason } }),
      { message: unlabeled(reason), messageKey: `${key}Unlabeled`, messageValues: { reason } }
    );
  }

  if (errorCode === 'removed') {
    return withRunLabel(
      queueItem,
      (label) => ({
        message: `“${label}” is no longer on the server queue; it may have been cleared.`,
        messageKey: 'notifications.runOutcome.removedMessage',
        messageValues: { label },
      }),
      {
        message: 'The run is no longer on the server queue; it may have been cleared.',
        messageKey: 'notifications.runOutcome.removedMessageUnlabeled',
      }
    );
  }

  if (errorCode === 'unexplained' || !reason) {
    return withRunLabel(
      queueItem,
      (label) => ({
        message: `“${label}” failed without a reason from the server.`,
        messageKey: 'notifications.runOutcome.unexplainedMessage',
        messageValues: { label },
      }),
      { message: 'The server gave no reason.', messageKey: 'notifications.runOutcome.unexplainedMessageUnlabeled' }
    );
  }

  return withRunLabel(
    queueItem,
    (label) => ({
      message: `“${label}”: ${reason}`,
      messageKey: 'notifications.runOutcome.failedMessage',
      messageValues: { label, reason },
    }),
    { message: reason }
  );
};

const createRunOutcomeNotification = (
  project: Project,
  // Absent when the run left the project's queue first, e.g. a clear; the notice then names the project alone.
  queueItem: QueueItem | undefined,
  {
    error,
    errorCode,
    failedBackendItemIds = [],
    status,
  }: {
    error?: string;
    errorCode?: QueueRunErrorCode;
    failedBackendItemIds?: readonly number[];
    status: 'cancelled' | 'failed';
  }
): WorkbenchNotification => {
  const titleValues = { project: project.name };

  if (status === 'cancelled') {
    return createNotification({
      ...describeRunLabel(queueItem),
      category: 'run-outcome',
      kind: 'info',
      projectId: project.id,
      title: `Run canceled in ${project.name}`,
      titleKey: 'notifications.runOutcome.canceled',
      titleValues,
    });
  }

  const [firstFailedItemId] = failedBackendItemIds;
  const itemCount = Math.max(queueItem?.backendItemIds?.length ?? 0, failedBackendItemIds.length);

  return createNotification({
    ...describeRunFailure(queueItem, error?.trim() ?? '', errorCode),
    ...(firstFailedItemId === undefined || errorCode === 'removed'
      ? {}
      : {
          actions: [
            { itemId: firstFailedItemId, kind: 'reveal-queue-item' },
            {
              itemIds: failedBackendItemIds,
              kind: 'retry-queue-item',
              ...(queueItem?.snapshot.destination === 'canvas' ? { canvasProjectId: project.id } : {}),
            },
          ],
        }),
    category: 'run-outcome',
    kind: 'error',
    persistent: true,
    projectId: project.id,
    ...(itemCount > 1 && failedBackendItemIds.length > 0
      ? {
          title: `${failedBackendItemIds.length} of ${itemCount} items failed in ${project.name}`,
          titleKey: 'notifications.runOutcome.failedItems',
          titleValues: { ...titleValues, count: failedBackendItemIds.length, total: itemCount },
        }
      : { title: `Run failed in ${project.name}`, titleKey: 'notifications.runOutcome.failed', titleValues }),
  });
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
      ...describeRunLabel(queueItem),
      category: 'enqueue',
      kind: 'success',
      projectId: after.id,
      title: `Queued in ${after.name}`,
      titleKey: 'notifications.run.queued',
      titleValues: { project: after.name },
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
  const slots = getCanvasStagingSlots(getCanvasWithPendingImages(project.canvas, pendingImages), project.queue.items);
  const selectedSlot = previousSelectedSlot ?? getSelectedCanvasStagingSlot(project);
  const preservedSlotIndex = findPreservedCanvasStagingSlotIndex(slots, selectedSlot);
  const { autoSwitchMode } = project.canvas.stagingArea;
  const newestIncoming = incomingImages.at(-1);

  if (newestIncoming && selectsLandedResult(autoSwitchMode, selectedSlot?.kind === 'candidate')) {
    const newestSlotIndex = getCanvasStagingCandidateSlotIndex(project, pendingImages, newestIncoming);

    if (newestSlotIndex !== -1) {
      return newestSlotIndex;
    }
  }

  return preservedSlotIndex !== -1
    ? preservedSlotIndex
    : clampStagedImageIndex(project.canvas.stagingArea.selectedImageIndex, slotCount);
};

const stageCanvasResultImages = (
  project: Project,
  queueItemId: string,
  images: GeneratedImageContract[],
  sourceBackendItemIds?: readonly (number | undefined)[],
  previousSelectedSlot?: CanvasStagingSlot
): Project => {
  const queueItem = project.queue.items.find((item) => item.id === queueItemId);

  if (!queueItem || queueItem.snapshot.canvas.documentRevision !== project.canvas.documentRevision) {
    return project;
  }

  return stageCanvasImages(
    project,
    queueItemId,
    queueItem.snapshot.canvas.document,
    images,
    sourceBackendItemIds,
    previousSelectedSlot
  );
};

const getRetriedCanvasResultDocument = (
  project: Project,
  queueItemId: string
): Pick<CanvasDocumentContractV3, 'bbox' | 'height' | 'width'> => {
  const queueItem = project.queue.items.find((item) => item.id === queueItemId);

  return queueItem?.snapshot.destination === 'canvas' &&
    queueItem.snapshot.canvas.documentRevision === project.canvas.documentRevision
    ? queueItem.snapshot.canvas.document
    : project.canvas.document;
};

const stageCanvasImages = (
  project: Project,
  queueItemId: string,
  queueDocument: Pick<CanvasDocumentContractV3, 'bbox' | 'height' | 'width'>,
  images: GeneratedImageContract[],
  sourceBackendItemIds?: readonly (number | undefined)[],
  previousSelectedSlot?: CanvasStagingSlot
): Project => {
  if (images.length === 0) {
    return project;
  }

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

type CanvasStagingFollow = NonNullable<WorkbenchState['canvasStagingFollow']>[string];

const NO_STAGING_FOLLOW: CanvasStagingFollow = { framedSlotIds: [], pinnedSlotId: null };

const updateCanvasStagingFollow = (
  state: WorkbenchState,
  projectId: string,
  update: (follow: CanvasStagingFollow) => CanvasStagingFollow
): WorkbenchState => {
  const previous = state.canvasStagingFollow?.[projectId] ?? NO_STAGING_FOLLOW;
  const next = update(previous);

  return next === previous
    ? state
    : { ...state, canvasStagingFollow: { ...state.canvasStagingFollow, [projectId]: next } };
};

const recordCanvasStagingFrame = (
  state: WorkbenchState,
  projectId: string,
  queueItemId: string,
  itemIndex: number
): WorkbenchState => {
  const project = state.projects.find((candidate) => candidate.id === projectId);
  const slot = project
    ? getCanvasStagingSlots(project.canvas, project.queue.items).find(
        (candidate) =>
          candidate.kind === 'placeholder' && candidate.queueItemId === queueItemId && candidate.itemIndex === itemIndex
      )
    : undefined;

  return slot
    ? updateCanvasStagingFollow(state, projectId, (follow) =>
        follow.framedSlotIds.includes(slot.id)
          ? follow
          : { ...follow, framedSlotIds: [...follow.framedSlotIds, slot.id] }
      )
    : state;
};

const updateCanvasStagingPin = (
  state: WorkbenchState,
  projectId: string,
  mutation: CanvasProjectMutation
): WorkbenchState => {
  const project = state.projects.find((candidate) => candidate.id === projectId);
  const isPick = mutation.type === 'setStagedImageIndex' || mutation.type === 'cycleStagedImage';

  if (!project || (!isPick && mutation.type !== 'setCanvasStagingAutoSwitch')) {
    return state;
  }

  const slot = isPick ? getSelectedCanvasStagingSlot(project) : undefined;
  const pinnedSlotId = slot ? pickStagedSlotAutoSwitch(slot).pinnedId : null;

  return updateCanvasStagingFollow(state, projectId, (follow) =>
    follow.pinnedSlotId === pinnedSlotId ? follow : { ...follow, pinnedSlotId }
  );
};

/** The reducer cannot tell waiting from running, so a pin holds until its placeholder is gone. */
const followCanvasStaging = (
  project: Project,
  slots: readonly CanvasStagingSlot[],
  follow: CanvasStagingFollow
): Project => {
  const { autoSwitchMode, selectedImageIndex } = project.canvas.stagingArea;

  if (autoSwitchMode === 'off') {
    return project;
  }

  const lives = slots.flatMap((slot) =>
    slot.kind === 'placeholder'
      ? [{ hasOwnFrame: follow.framedSlotIds.includes(slot.id), id: slot.id, state: 'running' as const }]
      : []
  );
  const followed = getAutoSwitchFollowedLive(autoSwitchMode, lives, follow.pinnedSlotId);
  const nextIndex = followed
    ? slots.findIndex((slot) => slot.id === followed.id)
    : slots[selectedImageIndex]?.kind === 'placeholder'
      ? getLatestCandidateSlotIndex(project, slots)
      : -1;

  return nextIndex === -1 || nextIndex === selectedImageIndex
    ? project
    : {
        ...project,
        canvas: { ...project.canvas, stagingArea: { ...project.canvas.stagingArea, selectedImageIndex: nextIndex } },
      };
};

/** Forgets frames and pins of placeholders that have gone, so a slot id a retry brings back starts fresh. */
const pruneCanvasStagingFollow = (
  follow: CanvasStagingFollow,
  slots: readonly CanvasStagingSlot[]
): CanvasStagingFollow => {
  const placeholderIds = new Set(slots.flatMap((slot) => (slot.kind === 'placeholder' ? [slot.id] : [])));
  const framedSlotIds = follow.framedSlotIds.filter((id) => placeholderIds.has(id));
  const pinnedSlotId =
    follow.pinnedSlotId !== null && placeholderIds.has(follow.pinnedSlotId) ? follow.pinnedSlotId : null;

  return framedSlotIds.length === follow.framedSlotIds.length && pinnedSlotId === follow.pinnedSlotId
    ? follow
    : { framedSlotIds, pinnedSlotId };
};

const reconcileCanvasStagingFollow = (previous: WorkbenchState, next: WorkbenchState): WorkbenchState => {
  if (next.projects === previous.projects && next.canvasStagingFollow === previous.canvasStagingFollow) {
    return next;
  }

  const previousById = new Map(previous.projects.map((project) => [project.id, project]));
  const follows: Record<string, CanvasStagingFollow> = {};
  let didChangeProjects = false;
  const projects = next.projects.map((project) => {
    const before = previousById.get(project.id);
    const follow = next.canvasStagingFollow?.[project.id];
    const isUnchanged =
      before !== undefined &&
      before.canvas.stagingArea === project.canvas.stagingArea &&
      before.canvas.documentRevision === project.canvas.documentRevision &&
      before.queue === project.queue &&
      follow === previous.canvasStagingFollow?.[project.id];

    if (isUnchanged) {
      if (follow) {
        follows[project.id] = follow;
      }
      return project;
    }

    const slots = getCanvasStagingSlots(project.canvas, project.queue.items);
    const pruned = pruneCanvasStagingFollow(follow ?? NO_STAGING_FOLLOW, slots);
    const followed = followCanvasStaging(project, slots, pruned);

    if (pruned.framedSlotIds.length > 0 || pruned.pinnedSlotId !== null) {
      follows[project.id] = pruned;
    }
    didChangeProjects ||= followed !== project;
    return followed;
  });

  const previousFollows = next.canvasStagingFollow ?? {};
  const followIds = Object.keys(follows);
  const didChangeFollows =
    followIds.length !== Object.keys(previousFollows).length ||
    followIds.some((projectId) => follows[projectId] !== previousFollows[projectId]);

  return didChangeFollows || didChangeProjects
    ? {
        ...next,
        ...(didChangeProjects ? { projects } : {}),
        ...(didChangeFollows ? { canvasStagingFollow: follows } : {}),
      }
    : next;
};

const getLatestCandidateSlotIndex = (project: Project, slots: readonly CanvasStagingSlot[]): number => {
  const latest = project.canvas.stagingArea.pendingImages.at(-1);

  return latest ? slots.findIndex((slot) => slot.kind === 'candidate' && slot.candidate === latest) : -1;
};

const clampCanvasStagingSelection = (project: Project, previousSelectedSlot?: CanvasStagingSlot): Project => {
  const slots = getCanvasStagingSlots(project.canvas, project.queue.items);
  const slotCount = slots.length;
  const preservedSlotIndex = findPreservedCanvasStagingSlotIndex(slots, previousSelectedSlot);
  // A followed slot that ends without a result hands over to the newest result, not the next placeholder.
  const fallbackIndex =
    project.canvas.stagingArea.autoSwitchMode === 'progress' && previousSelectedSlot?.kind === 'placeholder'
      ? getLatestCandidateSlotIndex(project, slots)
      : -1;
  const selectedImageIndex =
    preservedSlotIndex !== -1
      ? preservedSlotIndex
      : fallbackIndex !== -1
        ? fallbackIndex
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

const getPersistedPrimaryGalleryItemKey = (values: Record<string, unknown>): GalleryItemKey | null => {
  if (typeof values.selectedImageName === 'string') {
    return canonicalizeGalleryItemKey(values.selectedImageName);
  }

  const selectedItem = getSelectedGalleryItemFromValues(values);

  return selectedItem ? toGalleryItemKey(selectedItem) : null;
};

const getGalleryCompareState = (values: Record<string, unknown>): 'armed' | 'comparing' | null => {
  const compareItem = getGalleryItemFromPersistedValue(values, values.compareImage);
  const primaryKey = getPersistedPrimaryGalleryItemKey(values);

  if (!compareItem || primaryKey === null || parseGalleryItemKey(primaryKey).kind !== 'image') {
    return null;
  }

  return primaryKey === toGalleryItemKey(compareItem) ? 'armed' : 'comparing';
};

const getSelectionComparePatch = (values: Record<string, unknown>, item: GalleryItem): Record<string, unknown> => {
  if (item.kind === 'video') {
    return { compareImage: null };
  }

  const compareItem = getGalleryItemFromPersistedValue(values, values.compareImage);

  if (
    !compareItem ||
    toGalleryItemKey(compareItem) !== toGalleryItemKey(item) ||
    getGalleryCompareState(values) !== 'comparing'
  ) {
    return {};
  }

  const previousItem = getSelectedGalleryItemFromValues(values);

  return { compareImage: previousItem?.kind === 'image' ? previousItem : null };
};

/** Not a migration boundary: unknown/legacy input must run `loadCanvasState` first. */
const cloneCanvas = (canvas: CanvasStateContractV3): CanvasStateContractV3 => {
  const document = structuredClone(canvas.document);

  return {
    version: canvas.version,
    document,
    documentRevision: canvas.documentRevision,
    snapshots: canvas.snapshots.map((snapshot) => ({
      ...snapshot,
      document: structuredClone(snapshot.document),
      ...(snapshot.stagedImages
        ? { stagedImages: snapshot.stagedImages.map((image) => normalizeStagingCandidate(image, snapshot.document)) }
        : {}),
    })),
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

const settleWidgetValues = (
  typeId: WidgetTypeId,
  previous: Record<string, unknown>,
  values: Record<string, unknown>
): Record<string, unknown> => (typeId === 'gallery' ? settleGalleryValues(previous, values) : values);

const updateProjectWidgetValues = (
  project: Project,
  widgetId: WidgetTypeId,
  getValues: (values: Record<string, unknown>) => Record<string, unknown>
): Project =>
  updateProjectWidgetState(project, widgetId, (widgetState) => {
    const values = settleWidgetValues(widgetId, widgetState.values, getValues(widgetState.values));

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

  const values = settleWidgetValues(instance.typeId, instance.state.values, getValues(instance.state.values));

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

const cloneWidgetGraphs = (widgetGraphs: Project['widgetGraphs']): Project['widgetGraphs'] =>
  Object.fromEntries(Object.entries(widgetGraphs).map(([key, graph]) => [key, graph ? cloneGraph(graph) : graph]));

// Canvas pixel history belongs to the engine and graph history to each workflow; project undo preserves both.
const createUndoSnapshot = (project: Project): ProjectUndoSnapshot => ({
  invocation: { ...project.invocation },
  views: project.views,
  widgetGraphs: cloneWidgetGraphs(project.widgetGraphs),
  widgetInstances: cloneWidgetInstances(project.widgetInstances),
  workspaceId: project.workspaceId,
});

/**
 * Layout undo: the workspace, views and route revert; widget values stay, and no instance is lost. A project is never
 * put back on a workspace the account no longer has, nor given views of one.
 */
const restoreUndoSnapshot = (project: Project, snapshot: ProjectUndoSnapshot, account: AccountState): Project => {
  const views = Object.keys(snapshot.views).every((id) => doesLayoutPresetExist(account, id))
    ? snapshot.views
    : Object.fromEntries(Object.entries(snapshot.views).filter(([id]) => doesLayoutPresetExist(account, id)));

  return {
    ...project,
    invocation: { ...snapshot.invocation },
    views,
    widgetGraphs: { ...cloneWidgetGraphs(snapshot.widgetGraphs), ...project.widgetGraphs },
    widgetInstances: { ...cloneWidgetInstances(snapshot.widgetInstances), ...project.widgetInstances },
    workspaceId: doesLayoutPresetExist(account, snapshot.workspaceId) ? snapshot.workspaceId : project.workspaceId,
  };
};

type UndoWorkspace = NonNullable<UndoRedoEntry['workspace']>;

/** Edits sharing a `mergeKey` within the merge window fold into the step before them. */
const pushUndo = (
  project: Project,
  label: string,
  {
    customizeRoute,
    mergeKey,
    notice,
    workspace,
  }: {
    customizeRoute?: UndoRedoEntry['customizeRoute'];
    mergeKey?: string;
    notice?: LayoutUndoNotice;
    workspace?: UndoWorkspace;
  } = {}
): Project => {
  const previous = project.undoRedo.past.at(-1);
  const timestamp = now();

  // An undo in between (`future` non-empty) ends the burst: the state the user stood on stays its own step.
  if (
    mergeKey &&
    !workspace &&
    previous?.mergeKey === mergeKey &&
    project.undoRedo.future.length === 0 &&
    Date.parse(timestamp) - Date.parse(previous.mergedAt ?? previous.createdAt) <= LAYOUT_MERGE_WINDOW_MS
  ) {
    return {
      ...project,
      undoRedo: { future: [], past: [...project.undoRedo.past.slice(0, -1), { ...previous, mergedAt: timestamp }] },
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
          ...(notice ? { notice } : {}),
          project: createUndoSnapshot(project),
          ...(workspace ? { workspace } : {}),
          ...(customizeRoute ? { customizeRoute } : {}),
        },
      ].slice(-LAYOUT_HISTORY_LIMIT),
    },
  };
};

/** A step's change to a workspace in the account, recorded from the account before and after it. */
const toUndoWorkspace = (
  before: AccountState,
  after: AccountState,
  workspaceId: LayoutPresetId
): UndoWorkspace | undefined => {
  const recordBefore = readWorkspaceRecord(before, workspaceId);
  const recordAfter = readWorkspaceRecord(after, workspaceId);

  return isSameWorkspaceRecord(recordBefore, recordAfter)
    ? undefined
    : { after: recordAfter, before: recordBefore, workspaceId };
};

const LAYOUT_UNDO_STALE_TITLE_KEY = 'shell.layoutUndo.workspaceChangedTitle';
const LAYOUT_UNDO_STALE_MESSAGE_KEY = 'shell.layoutUndo.workspaceChangedMessage';

/**
 * One undo or redo step of a project's layout history. A step that changed a workspace in the account puts that
 * workspace back only while the account still holds what the step left; when something changed it since (another
 * project, Customize, a deleted workspace), the project's own view and route still step back, and a notice says why
 * the workspace did not.
 */
const stepProjectHistory = (state: WorkbenchState, projectId: string, direction: 'undo' | 'redo'): WorkbenchState => {
  const project = state.projects.find((candidate) => candidate.id === projectId);
  const entry = direction === 'undo' ? project?.undoRedo.past.at(-1) : project?.undoRedo.future[0];

  if (!project || !entry) {
    return state;
  }

  const change = entry.workspace;
  const isCurrent =
    change !== undefined && isSameWorkspaceRecord(readWorkspaceRecord(state.account, change.workspaceId), change.after);
  const account =
    change && isCurrent ? writeWorkspaceRecord(state.account, change.workspaceId, change.before) : state.account;
  // Done's route moves only with its workspace, so Invoke never aims at a source the workspace no longer places.
  const routeMoved =
    entry.customizeRoute && isCurrent ? stepRouteMove(project.invocation, entry.customizeRoute.move) : null;
  const opposite: UndoRedoEntry = {
    createdAt: now(),
    id: createId(direction === 'undo' ? 'redo' : 'undo'),
    label: entry.label,
    project: createUndoSnapshot(project),
    ...(change && isCurrent
      ? { workspace: { after: change.before, before: change.after, workspaceId: change.workspaceId } }
      : {}),
    ...(entry.customizeRoute
      ? {
          customizeRoute:
            routeMoved && entry.customizeRoute.move
              ? { move: { from: entry.customizeRoute.move.to, to: entry.customizeRoute.move.from } }
              : {},
        }
      : {}),
  };
  const undoRedo =
    direction === 'undo'
      ? {
          future: [opposite, ...project.undoRedo.future].slice(0, LAYOUT_HISTORY_LIMIT),
          past: project.undoRedo.past.slice(0, -1),
        }
      : {
          future: project.undoRedo.future.slice(1),
          past: [...project.undoRedo.past, opposite].slice(-LAYOUT_HISTORY_LIMIT),
        };
  const restored = restoreUndoSnapshot(project, entry.project, account);
  const stepped = replaceProject(
    { ...state, account },
    {
      ...restored,
      ...(entry.customizeRoute ? { invocation: routeMoved ?? project.invocation } : {}),
      undoRedo,
    }
  );

  return change && !isCurrent
    ? addNotification(
        stepped,
        createNotification({
          kind: 'info',
          message: 'The workspace changed after this step, so only how this project looks at it was put back.',
          messageKey: LAYOUT_UNDO_STALE_MESSAGE_KEY,
          projectId,
          title: 'Workspace not changed back',
          titleKey: LAYOUT_UNDO_STALE_TITLE_KEY,
        })
      )
    : stepped;
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

export const loadWorkbenchProject = (raw: Project): ProjectLoadResult => {
  const refused = gateProjectCanvases(raw);

  return refused ? { refused, status: 'refused' } : { project: normalizeWorkbenchProject(raw), status: 'loaded' };
};

/** A canvas that fails to reload is kept as-is, never rewritten. */
export const normalizeWorkbenchProject = (
  project: Project,
  options: {
    /** Drop session-only infinite-window anchors on imported/server documents. */
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
    floatingWidgets: storedFloatingWidgets,
    graphHistory: _graphHistory,
    heldView: storedHeldView,
    importedLayout: storedImportedLayout,
    layout: storedLayout,
    droppedArrangements: storedDroppedArrangements,
    legacyLayout: carriedLegacyLayout,
    presetWorkingLayouts: storedWorkingLayouts,
    projectGraph: legacyProjectGraph,
    recoveredAt: _recoveredAt,
    recoveryOf: _recoveryOf,
    views: storedViews,
    widgetRegions: storedWidgetRegions,
    workspaceId: storedWorkspaceId,
    ...persistentProject
  } = project as Project &
    StoredLayoutFields & {
      graphHistory?: unknown;
      projectGraph?: unknown;
      recoveredAt?: unknown;
      recoveryOf?: unknown;
    };
  const instances = cloneWidgetInstances(project.widgetInstances ?? createWidgetInstances());
  // A document names its workspace from the split on; one without that and with a stored layout predates it.
  const stored =
    carriedLegacyLayout || typeof storedWorkspaceId === 'string' || !isRecordValue(storedLayout)
      ? null
      : loadStoredProjectLayout(
          {
            floatingWidgets: storedFloatingWidgets,
            heldView: storedHeldView,
            importedLayout: storedImportedLayout,
            layout: storedLayout,
            presetWorkingLayouts: storedWorkingLayouts,
            widgetRegions: storedWidgetRegions,
          },
          instances,
          (typeId) => createWidgetInstance(typeId)
        );
  const widgetInstances = stored?.widgetInstances ?? withoutRetiredTypes(instances);
  const legacyLayout = carriedLegacyLayout ?? stored?.legacyLayout;

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

  for (const [instanceId, instance] of Object.entries(widgetInstances)) {
    if (instance.typeId !== 'gallery') {
      continue;
    }

    // Drop rankings unresolvable in this session, as serialization does, so hydration matches the sync baseline.
    const strippedSearchValues = stripUnresolvableGallerySearch(instance.state.values);
    const strippedValues = isArriving
      ? (stripInfiniteWindowAnchor(strippedSearchValues ?? instance.state.values) ?? strippedSearchValues)
      : strippedSearchValues;
    const hasRecentImages = 'recentImages' in instance.state.values;
    // The retired live-follow pause stamp; it leaves stored workspaces on the next save.
    const hasRetiredPause = 'liveFollowPausedAt' in instance.state.values;
    const settledValues = settleGalleryValues(instance.state.values, strippedValues ?? instance.state.values);

    if (settledValues === instance.state.values && !hasRecentImages && !hasRetiredPause) {
      continue;
    }

    const { liveFollowPausedAt: _retiredPause, ...values } = settledValues;

    widgetInstances[instanceId] = {
      ...instance,
      state: {
        ...instance.state,
        values: hasRecentImages ? { ...values, recentImages: getBoundedRecentImages(values.recentImages) } : values,
      },
    };
  }

  return {
    ...persistentProject,
    canvas,
    events: isArriving ? [] : project.events.slice(0, PROJECT_EVENT_LIMIT),
    lastFloatingGeometry: normalizeLastFloatingGeometry(
      (project as Partial<Project>).lastFloatingGeometry,
      (instanceId) => Object.hasOwn(widgetInstances, instanceId),
      legacyLayout?.floatingWidgets
    ),
    ...(legacyLayout ? { legacyLayout } : {}),
    ...(() => {
      // Untrusted on load; workspaces the account lacks go once it is known (`settleArrivingProject`).
      const dropped = Array.isArray(storedDroppedArrangements)
        ? [...new Set(storedDroppedArrangements.filter((id): id is string => typeof id === 'string' && id.length > 0))]
        : [];

      return dropped.length > 0 ? { droppedArrangements: dropped } : {};
    })(),
    promptHistory: normalizePromptHistory((project as Partial<Project>).promptHistory),
    queue: isArriving ? { items: [] } : project.queue,
    settings: normalizeProjectSettings(project.settings),
    // Views of workspaces the account lacks go once the account is known (`settleProjectLayout`).
    views: normalizeViews(storedViews, () => true),
    widgetInstances,
    workflowHistories: isArriving ? {} : (project.workflowHistories ?? {}),
    workflows: resolveProjectWorkflows(project.workflows, legacyProjectGraph),
    workspaceId:
      typeof storedWorkspaceId === 'string' && storedWorkspaceId.length > 0
        ? resolveLayoutPresetId(storedWorkspaceId)
        : (legacyLayout?.layout.presetId ?? defaultLayoutPreset.id),
  };
};

/** Session snapshots bypass `migrateProjectDocument`: without a usable collection they get one blank workflow. */
const resolveProjectWorkflows = (
  candidate: ProjectWorkflowCollection | undefined,
  legacyProjectGraph: unknown
): ProjectWorkflowCollection =>
  normalizeProjectWorkflowCollection(candidate) ??
  migrateProjectGraphToCollection(legacyProjectGraph ?? {}) ??
  createProjectWorkflowCollection(createBlankWorkflowDocument());

export const withAuthoritativeProjectBoard = (project: Project, boardId: string): Project =>
  updateProjectWidgetValues(project, 'gallery', (values) =>
    values.projectBoardId === boardId ? values : { ...values, projectBoardId: boardId }
  );

const createCanvasState = (): CanvasStateContractV3 => createNewCanvasState();

const WORKSPACE_SOURCES = { getNaturalDestination, getSourceId: getAvailableSourceIdForWidgetTypeId };

// Its first activation enters the workspace (fronts); until then only the route needs the job.
const createProject = (index: number, id: string, preset: LayoutPreset): Project =>
  ensureLayoutInstances(
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
      invocation: { ...defaultInvocationRoute, ...getWorkspaceJob(preset, preset.snapshot, WORKSPACE_SOURCES) },
      name: `Project Name #${index}`,
      promptHistory: [],
      queue: { items: [] },
      // A new project starts on Auto-switch In progress (only a stored one lacks the mode) and pins no flag.
      settings: normalizeProjectSettings({ viewerAutoSwitch: BUILT_IN_PROJECT_SETTINGS.viewerAutoSwitch }),
      undoRedo: { future: [], past: [] },
      views: {},
      widgetGraphs: {},
      widgetInstances: createWidgetInstances(),
      workflowHistories: {},
      workflows: createProjectWorkflowCollection(createBlankWorkflowDocument()),
      workspaceId: preset.id,
    },
    preset.snapshot
  );

const getNextProjectIndex = (projects: Project[]): number => {
  const usedIndices = projects.map((project) => Number(project.name.match(/#(\d+)$/)?.[1] ?? 0));

  return Math.max(0, ...usedIndices) + 1;
};

/** Project ids are global across accounts, so a draft takes a UUID one like every other new project. */
export const createDraftProject = (projects: Project[], account?: WorkbenchState['account']): Project =>
  createProject(
    getNextProjectIndex(projects),
    createProjectId(),
    account ? resolveSavedLayoutPreset(normalizeWorkbenchAccount(account), defaultLayoutPreset.id) : defaultLayoutPreset
  );

const updateActiveProject = (state: WorkbenchState, getProject: (project: Project) => Project): WorkbenchState =>
  updateProjectById(state, state.activeProjectId, getProject);

const replaceProject = (state: WorkbenchState, project: Project): WorkbenchState =>
  updateProjectById(state, project.id, () => project);

export const resolvePanelToggle = (
  widgetRegions: Record<WidgetRegion, Pick<WidgetRegionState, 'instanceIds' | 'isCollapsed'>>,
  regions: readonly WidgetRegion[]
): { regions: WidgetRegion[]; shouldCollapse: boolean } => {
  const occupied = regions.filter((region) => widgetRegions[region].instanceIds.length > 0);
  return { regions: occupied, shouldCollapse: occupied.some((region) => !widgetRegions[region].isCollapsed) };
};

// Layout: what a project shows is its workspace's structure (the account's, or its Customize draft) seen through its
// view of that workspace. View actions write the view; structure actions write the structure, then compose view.

const getLayout = (state: LayoutSource, project: Project, context: WorkbenchReducerContext): ResolvedLayout =>
  resolveLayout(getProjectStructure(state, project), getProjectView(state, project), context.availability);

/** Gives every instance a layout places a record of values; existing values are never touched. */
const ensureLayoutInstances = (
  project: Project,
  layout: { widgetInstances: Readonly<Record<WidgetInstanceId, { id: WidgetInstanceId; typeId: WidgetTypeId }>> }
): Project => {
  const missing = Object.values(layout.widgetInstances).filter(
    (instance) => !Object.hasOwn(project.widgetInstances, instance.id)
  );

  return missing.length === 0
    ? project
    : {
        ...project,
        widgetInstances: {
          ...project.widgetInstances,
          ...Object.fromEntries(missing.map(({ id, typeId }) => [id, createWidgetInstance(typeId, id)])),
        },
      };
};

/** Writes the project's view of its workspace; the same project when the view does not change. */
const updateProjectView = (
  state: LayoutSource,
  project: Project,
  update: (view: WorkspaceView | undefined, structure: WorkspaceStructure) => WorkspaceView | undefined
): Project => {
  const view = getProjectView(state, project);
  const next = update(view, getProjectStructure(state, project));

  return next === view
    ? project
    : { ...project, views: withView(project.views, getProjectViewId(state, project), next) };
};

/** Fronts a member of the region and shows the region: what opening, docking and moving a widget there end with. */
const showInRegion = (
  state: LayoutSource,
  project: Project,
  region: WidgetRegion,
  instanceId: WidgetInstanceId,
  availability: LayoutAvailability
): Project =>
  updateProjectView(state, project, (view, structure) =>
    setRegionCollapsed(frontRegion(view, structure, region, instanceId, availability), structure, region, false)
  );

/**
 * A structural gesture on a project's workspace: the Customize draft while the project customizes, else (rearranging
 * without Customize) the workspace in the account, for every project on it. `compose` then writes the view effects
 * the gesture has for this project over the edited structure. Windows that close are remembered, instances the
 * structure places get values, and view entries the structure no longer has go.
 */
const applyStructureGesture = (
  state: WorkbenchState,
  project: Project,
  action: StructureAction,
  context: WorkbenchReducerContext,
  {
    compose = (next) => next,
    notice,
  }: {
    compose?: (project: Project, source: LayoutSource) => Project;
    notice?: { label: string; value: LayoutUndoNotice };
  } = {}
): WorkbenchState => {
  const session = getCustomizeSession(state.customizing, project);
  const before = getProjectStructure(state, project);
  const after = reduceStructure(before, action, context.availability);

  if (after === before) {
    return state;
  }

  const edited: WorkbenchState = session
    ? { ...state, customizing: { ...state.customizing, [project.id]: { ...session, structure: after } } }
    : { ...state, account: writeWorkspaceStructure(state.account, project.workspaceId, after) };
  const stepped =
    notice && !session
      ? pushUndo(project, notice.label, {
          notice: notice.value,
          workspace: toUndoWorkspace(state.account, edited.account, project.workspaceId),
        })
      : project;
  const pruned = updateProjectView(edited, ensureLayoutInstances(stepped, after), (view) =>
    pruneView(view, after, context.availability)
  );
  const composed = compose(pruned, edited);

  return replaceProject(
    edited,
    rememberLayoutWindows(composed, getLayout(state, project, context), getLayout(edited, composed, context))
  );
};

/** A window that closed or docked keeps its last geometry, so floating it again reopens it there. */
const rememberLayoutWindows = (project: Project, before: ResolvedLayout, after: ResolvedLayout): Project => {
  const lastFloatingGeometry = rememberClosedWindows(
    project.lastFloatingGeometry,
    before.floatingWidgets,
    after.floatingWidgets,
    (instanceId) => Object.hasOwn(project.widgetInstances, instanceId)
  );

  return lastFloatingGeometry === project.lastFloatingGeometry ? project : { ...project, lastFloatingGeometry };
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

    if (typeof record.id !== 'string' || typeof record.label !== 'string' || !isStoredStructure(record.snapshot)) {
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
        snapshot: readStoredStructure(record.snapshot),
      },
    ];
  });
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

      const structure =
        isBuiltInLayoutPresetId(resolvedPresetId) && isStoredStructure(snapshot)
          ? readStoredOverride(resolvedPresetId, snapshot)
          : undefined;

      return structure ? [[resolvedPresetId, structure]] : [];
    })
  );
};

export const normalizeWorkbenchAccount = (value: unknown): WorkbenchState['account'] => {
  const account = value && typeof value === 'object' ? (value as Partial<WorkbenchState['account']>) : undefined;
  const customLayoutPresets = normalizeCustomLayoutPresets(account?.customLayoutPresets);
  const layoutPresetOverrides = normalizeLayoutPresetOverrides(account?.layoutPresetOverrides);
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
    layoutPresetOverrides,
    layoutPresetRouteOverrides: normalizeLayoutPresetRouteOverrides(account?.layoutPresetRouteOverrides),
    // An account with nothing stored is new: it has no stored arrangements to adopt.
    ...(account === undefined || account.workspaceLayoutsAdopted === true ? { workspaceLayoutsAdopted: true } : {}),
    ...(() => {
      // Listed only while a stored structure carries the mark: an override read as shipped is gone.
      const written = readWrittenStructureIds(account as Record<string, unknown> | undefined)
        .map(resolveLayoutPresetId)
        .filter((id) =>
          isBuiltInLayoutPresetId(id)
            ? layoutPresetOverrides[id] !== undefined
            : customLayoutPresets.some((preset) => preset.id === id)
        );

      return written.length > 0 ? { writtenStructureIds: [...new Set(written)] } : {};
    })(),
  };
};

/**
 * A project arriving after hydration, settled against the account (`settleArrivingProject`). When it adopts, the
 * account changes under every open project, so each gets values for what its workspace now places.
 */
const settleArriving = (
  state: WorkbenchState,
  project: Project,
  context: WorkbenchReducerContext
): { project: Project; state: WorkbenchState } => {
  const settled = settleArrivingProject(state.account, project, {
    adoptionPending: state.layoutAdoptionPending === true,
    surfaces: context.availability,
  });
  const next: WorkbenchState =
    settled.account === state.account
      ? state
      : {
          ...state,
          account: settled.account,
          layoutAdoptionPending: true,
          projects: state.projects.map((candidate) => withWorkspaceInstances(settled.account, candidate)),
        };

  return { project: withWorkspaceInstances(settled.account, settled.project), state: next };
};

/** A settled project with values for every instance its workspace places. */
const withWorkspaceInstances = (account: AccountState, project: Project): Project =>
  ensureLayoutInstances(project, getWorkspaceStructure(account, project.workspaceId));

const normalizeWorkbenchState = (state: WorkbenchState, context: WorkbenchReducerContext): WorkbenchState => {
  // Built explicitly: legacy snapshots carried preferences inside the account, which must not resurface.
  const normalizedAccount = normalizeWorkbenchAccount(state.account);
  const restored = state.projects.map((project) => normalizeWorkbenchProject(project));
  // Projectless snapshots (e.g. after failed canvas recovery) still need an active project.
  const loaded = restored.length > 0 ? restored : [createDraftProject([], normalizedAccount)];
  const activeProjectId = loaded.some((project) => project.id === state.activeProjectId)
    ? state.activeProjectId
    : loaded[0]!.id;
  const { layoutAdoptionPending: _pending, ...rest } = state;
  const settled = settleHydratedProjects(normalizedAccount, loaded, activeProjectId, context.availability);

  return {
    ...rest,
    account: settled.account,
    activeProjectId,
    backendConnection: { status: 'connecting' },
    ...(settled.adoptionPending ? { layoutAdoptionPending: true } : {}),
    notifications: [],
    projects: settled.projects.map((project) => withWorkspaceInstances(settled.account, project)),
  };
};

/** What each one-time arrangement notice says, by the project field that holds the workspaces it names. */
const ARRANGEMENT_NOTICES = {
  adoptedArrangements: {
    message: (workspaces: string, count: number) =>
      count === 1
        ? `${workspaces} now uses this project's arrangement in every project.`
        : `${workspaces} now use this project's arrangement in every project.`,
    messageKey: 'shell.workspace.arrangementAdoptedMessage',
    title: 'Workspaces updated',
    titleKey: 'shell.workspace.arrangementAdoptedTitle',
  },
  droppedArrangements: {
    message: (workspaces: string, count: number) =>
      count === 1
        ? `Workspaces now look the same in every project, so this project's own arrangement of ${workspaces} was not kept. Customize the workspace to arrange it again.`
        : `Workspaces now look the same in every project, so this project's own arrangements of ${workspaces} were not kept. Customize a workspace to arrange it again.`,
    messageKey: 'shell.workspace.arrangementDroppedMessage',
    title: 'Workspaces are now shared',
    titleKey: 'shell.workspace.arrangementDroppedTitle',
  },
} as const;

/**
 * Tells, once, what happened to the active project's own arrangements when its stored layout migrated: dropped, or
 * (adopting after hydration) taken by its workspaces, offering to customize the first of them. A drop is stored with
 * the project, so a reload before it is told keeps it.
 */
const announceDroppedArrangements = (state: WorkbenchState): WorkbenchState => {
  const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

  if (!project || (!project.droppedArrangements && !project.adoptedArrangements)) {
    return state;
  }

  const { adoptedArrangements: _adopted, droppedArrangements: _dropped, ...told } = project;
  let next = replaceProject(state, told);

  for (const field of ['adoptedArrangements', 'droppedArrangements'] as const) {
    const ids = project[field];

    if (!ids) {
      continue;
    }

    const notice = ARRANGEMENT_NOTICES[field];
    const labels = ids.map((id) => {
      const { label, labelKey } = resolveSavedLayoutPreset(state.account, id);

      return labelKey === undefined ? { text: label } : { key: labelKey, text: label };
    });
    const workspaces = labels.map(({ text }) => text).join(', ');

    next = addNotification(
      next,
      createNotification({
        actions: [{ kind: 'customize-workspace', workspaceId: ids[0]! }],
        kind: 'info',
        message: notice.message(workspaces, ids.length),
        messageKey: notice.messageKey,
        messageListValues: { workspaces: labels },
        messageValues: { count: ids.length, workspaces },
        projectId: project.id,
        title: notice.title,
        titleKey: notice.titleKey,
      })
    );
  }

  return next;
};

const getAvailableLayoutPreset = (state: WorkbenchState, presetId: LayoutPresetId): LayoutPreset =>
  resolveSavedLayoutPreset(state.account, presetId);

/** Applies the entry rule to the project's view of the workspace it is on: view and route only. */
const enterProjectWorkspace = (state: LayoutSource, project: Project, context: WorkbenchReducerContext): Project => {
  const structure = getProjectStructure(state, project);
  const view = getProjectView(state, project);
  const entry = enterWorkspace({
    invocation: project.invocation,
    resolve: (candidate) => resolveLayout(structure, candidate, context.availability),
    sources: WORKSPACE_SOURCES,
    structure,
    surfaces: context.availability,
    view,
    workspace: getProjectWorkspace(state, project),
  });

  return entry.view === view && entry.invocation === project.invocation
    ? project
    : {
        ...project,
        invocation: entry.invocation,
        views: withView(project.views, getProjectViewId(state, project), entry.view),
      };
};

/** Going home: the entry rule over the active project's own view, with no undo step. */
const enterActiveWorkspace = (state: WorkbenchState, context: WorkbenchReducerContext): WorkbenchState =>
  updateActiveProject(state, (project) => enterProjectWorkspace(state, project, context));

/** A project's first activation in this editor session is an entry; returning to one already entered is not. */
const enterArrivingProject = (state: WorkbenchState, context: WorkbenchReducerContext): WorkbenchState => {
  const enteredProjectIds = state.enteredProjectIds ?? [];

  return enteredProjectIds.includes(state.activeProjectId)
    ? state
    : { ...enterActiveWorkspace(state, context), enteredProjectIds: [...enteredProjectIds, state.activeProjectId] };
};

/** Customize starts and ends on a fresh history, so no undo crosses its Done or Cancel to change structure. */
const withoutLayoutHistory = (project: Project): Project =>
  project.undoRedo.past.length === 0 && project.undoRedo.future.length === 0
    ? project
    : { ...project, undoRedo: { future: [], past: [] } };

/** Moves the active project onto the workspace and enters it, as one undoable step. */
const switchActiveProjectLayoutPreset = (
  state: WorkbenchState,
  preset: LayoutPreset,
  context: WorkbenchReducerContext
): WorkbenchState =>
  updateActiveProject(state, (project) => {
    const nextProject = pushUndo(project, 'Update layout', { notice: { kind: 'switchPreset', presetId: preset.id } });

    return {
      ...enterProjectWorkspace(state, { ...nextProject, workspaceId: preset.id }, context),
      events: prependProjectEvent(nextProject.events, {
        createdAt: now(),
        id: createId('event'),
        summary: 'Updated active layout',
        type: 'layout-updated',
      }),
    };
  });

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

const applyAutoRouteForRevealedInstance = (
  project: Project,
  instanceId: WidgetInstanceId,
  context: WorkbenchReducerContext
): Project => applyAutoRouteForWidgetReveal(project, project.widgetInstances[instanceId]?.typeId, context);

/** Auto-routes to what a region newly shows; `previousRegion` and `nextRegion` are as laid out before and after. */
const applyAutoRouteForRegionFront = (
  project: Project,
  previousRegion: WidgetRegionState,
  nextRegion: WidgetRegionState,
  context: WorkbenchReducerContext
): Project => {
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
  defaults: ScopedDefaults,
  route: InvocationRoute,
  runtime: InvocationRuntimeInputs,
  models?: readonly ModelConfig[],
  workflowGenerators?: WorkflowGeneratorResolutions,
  workflowDocument: ProjectGraphState = getActiveProjectGraph(project),
  upscaleSourceMetadata?: UpscaleSourceMetadata
): {
  graph: GraphContract;
  widgetStates: WidgetStateMap;
  workflowJson?: Record<string, unknown>;
  workflow?: Omit<WorkflowSubmissionPlan, 'graph'>;
  projectWorkflowId?: string;
} | null => {
  const widgetStates = getWidgetStatesSnapshot(project.widgetInstances);
  const projectSettings = getEffectiveProjectSettings(project.settings, defaults[PROJECT_SETTINGS_OWNER]);
  const randDevice = projectSettings.useCpuNoise ? 'cpu' : runtime.noiseDevice;

  if (route.sourceId === 'workflow') {
    const templatesSnapshot = runtime.templates;

    if (templatesSnapshot.status !== 'loaded') {
      return null;
    }

    const plan = planWorkflowSubmission(workflowDocument, templatesSnapshot.templates, {
      autoBoardId: getGalleryAutoAddBoardId(
        getResolvedProjectWidgetValues(project, 'gallery', defaults),
        (boardId: string) => runtime.boardDestinationIds.includes(boardId)
      ),
      batchCount: sanitizeBatchCount(widgetStates.workflow?.values.batchCount),
      generators: workflowGenerators,
      isBoardDestination: (boardId) => runtime.boardDestinationIds.includes(boardId),
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
    const currentValues = resolveUpscaleSubmissionValues(syncedValues, {
      promptDraft: getPromptDraftFromValues(getProjectWidgetValues(project, 'generate')),
      sourceImageName: upscaleSourceMetadata?.imageName,
    });

    if (!currentValues || getUpscaleValidationReasons(currentValues, models).length > 0) {
      return null;
    }

    const resolvedValues: UpscaleWidgetValues = { ...currentValues, seed: resolveUpscaleSeed(currentValues) };
    const compiledGraph = compileUpscaleGraph(
      resolvedValues,
      route.destination,
      projectSettings,
      randDevice,
      upscaleSourceMetadata?.metadata ?? null
    ).graph;

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

    // Video's prompt is deliberately independent of the draft Generate and Upscale share.
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
    projectSettings,
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
            paginationMode: getGallerySettings(applyWidgetDefaults(values, 'gallery', state.scopedDefaults))
              .paginationMode,
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
  removedItemKeys: ReadonlySet<GalleryItemKey>,
  successorProjectId: string | null = null
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
    const keepComparisonForSuccessor = project.id === successorProjectId;
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
      const nextSelectedImageName = selectedNameKey && removedItemKeys.has(selectedNameKey) ? null : selectedImageName;
      const nextCompareImage =
        (compareImageKey && removedItemKeys.has(compareImageKey)) ||
        (!keepComparisonForSuccessor && !nextSelectedImage && !nextSelectedImageName)
          ? null
          : compareImage;

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

    // Sweep raw slots before normalization so excluded media references cannot survive deletion.
    const withoutVideoMedia = updateProjectWidgetValues(withoutUpscaleInput, 'video', (rawValues) =>
      clearDeletedVideoMedia(rawValues, removedImageNames, removedVideoNames)
    );

    didChange ||= withoutVideoMedia !== project;
    return withoutVideoMedia;
  });

  return didChange ? { ...state, projects } : state;
};

/** Its own inbox is left out: it exists only with the project and goes with it. */
export const getProjectGalleryBoardReferences = (project: Project): string[] => {
  const values = getWidgetValues(project, 'gallery');
  const autoAddBoardId = getGalleryChosenAutoAddBoardId(values);
  const references = new Set<string>();

  for (const boardId of [
    values.selectedBoardId,
    autoAddBoardId,
    ...project.queue.items
      .filter((item) => item.status === 'pending' || item.status === 'running')
      .map((item) => item.snapshot.galleryBoardId),
  ]) {
    if (typeof boardId === 'string' && boardId !== values.projectBoardId) {
      references.add(boardId);
    }
  }

  return [...references];
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
      ...(autoAddBoardWasDeleted ? { autoAddBoardId: null } : {}),
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

const updateGalleryValuesAndStopAutoSwitch = (
  state: WorkbenchState,
  getValues: (values: Record<string, unknown>) => Record<string, unknown>,
  projectId = state.activeProjectId
): WorkbenchState =>
  updateProjectById(state, projectId, (project) => {
    const { mode } = pickAutoSwitch({ kind: 'result' });

    return updateProjectWidgetValues(
      project.settings.viewerAutoSwitch === mode
        ? project
        : { ...project, settings: { ...project.settings, viewerAutoSwitch: mode } },
      'gallery',
      getValues
    );
  });

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

const mergeImageResults = (
  existingImages: GeneratedImageContract[] | undefined,
  incomingImages: GeneratedImageContract[]
): GeneratedImageContract[] => {
  const existing = existingImages ?? [];
  const existingNames = new Set(existing.map((image) => image.imageName));

  return [...existing, ...incomingImages.filter((image) => !existingNames.has(image.imageName))];
};

/** A run already active, or a Canvas run whose document was since replaced, stays finished. */
const reopenQueueItemForRetry = (
  project: Project,
  queueItemId: string,
  retriedItems: readonly { itemId: number; sourceItemId: number }[]
): Project => {
  const queueItem = project.queue.items.find((item) => item.id === queueItemId);
  const ownedItemIds = new Set([...(queueItem?.backendItemIds ?? []), ...(queueItem?.retriedBackendItemIds ?? [])]);
  const itemIds = retriedItems.filter(({ sourceItemId }) => ownedItemIds.has(sourceItemId)).map(({ itemId }) => itemId);

  if (
    !queueItem ||
    itemIds.length === 0 ||
    queueItem.cancellationPending === true ||
    (queueItem.status !== 'completed' && queueItem.status !== 'failed' && queueItem.status !== 'cancelled') ||
    (queueItem.snapshot.destination === 'canvas' &&
      queueItem.snapshot.canvas.documentRevision !== project.canvas.documentRevision)
  ) {
    return project;
  }

  return updateQueueItem(project, queueItemId, (item) => {
    const {
      cancelledBackendItemIds: _cancelled,
      completedBackendItemIds: _completed,
      error: _error,
      localRecoveryState: _localRecoveryState,
      ...reopened
    } = item;

    return {
      ...reopened,
      backendItemIds: itemIds,
      retriedBackendItemIds: [...ownedItemIds],
      status: 'running',
    };
  });
};

const isSameSubmissionRetry = (left: QueueSubmissionRetry | undefined, right: QueueSubmissionRetry): boolean =>
  left !== undefined &&
  left.reason === right.reason &&
  left.failureCount === right.failureCount &&
  left.serverErrorCount === right.serverErrorCount &&
  left.serverErrorSince === right.serverErrorSince &&
  left.status === right.status &&
  left.message === right.message &&
  left.failedAt === right.failedAt;

const withoutSubmissionRetry = (item: QueueItem): QueueItem => {
  if (item.submissionRetry === undefined) {
    return item;
  }
  const { submissionRetry: _, ...rest } = item;
  return rest;
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

const shouldSelectGalleryResult = (project: Project, galleryValues: Record<string, unknown>): boolean =>
  selectsLandedResult(project.settings.viewerAutoSwitch, typeof galleryValues.selectedImageName === 'string');

const getGalleryResultSelectionValues = (
  galleryValues: Record<string, unknown>,
  item: GalleryItem,
  defaults: ScopedDefaults
): Record<string, unknown> => {
  const itemKey = toGalleryItemKey(item);
  const gallerySettings = getGallerySettings(applyWidgetDefaults(galleryValues, 'gallery', defaults));
  const mediaKind = toGalleryMediaKind(gallerySettings.mediaFilter);

  return {
    // A comparison waiting for its second image is the user's to complete; a result arriving does not complete it.
    ...(item.kind === 'video' || getGalleryCompareState(galleryValues) === 'armed' ? { compareImage: null } : {}),
    selectedImage: item,
    selectedImageName: itemKey,
    selectedImageNames: [itemKey],
    selectedImagePage: 0,
    selectedImageQuery: {
      boardId: item.boardId,
      // No view lists a Canvas-owned (`other`) item, so no listing can step from it: keep the one on screen.
      galleryView: getGalleryItemView(item) ?? getGalleryView(galleryValues),
      imageOrderDir: gallerySettings.imageOrderDir,
      ...(mediaKind === item.kind ? { mediaKind } : {}),
      page: 0,
      paginationMode: gallerySettings.paginationMode,
      searchTerm: '',
      starredOnly: false,
    },
  };
};

const updateGalleryWithResultImages = (
  project: Project,
  images: GeneratedImageContract[],
  defaults: ScopedDefaults
): Project => {
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
    newImages[0] && shouldSelectGalleryResult(project, galleryValues) ? newImages[0] : undefined;
  return updateProjectWidgetValues(project, 'gallery', () => ({
    ...galleryValues,
    recentImages: getBoundedRecentImages([...newImages, ...previousImages]),
    ...(nextSelectedImage
      ? getGalleryResultSelectionValues(galleryValues, legacyGeneratedImageToGalleryItem(nextSelectedImage), defaults)
      : { selectedImageNames: getPersistedSelectedGalleryItemKeys(galleryValues) }),
  }));
};

/** Only the newest video not routed before takes the selection, or a repeat pass would take it back. */
const routeGalleryResultVideos = (
  project: Project,
  queueItemId: string,
  videos: readonly GeneratedVideoContract[],
  defaults: ScopedDefaults,
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
  if (!shouldSelectGalleryResult(policyProject, getWidgetValues(policyProject, 'gallery'))) {
    return nextProject;
  }

  const galleryValues = getWidgetValues(nextProject, 'gallery');

  return updateProjectWidgetValues(nextProject, 'gallery', () => ({
    ...galleryValues,
    ...getGalleryResultSelectionValues(galleryValues, generatedVideoToGalleryItem(video), defaults),
  }));
};

const routeQueueItemPartialResults = (
  project: Project,
  queueItemId: string,
  backendItemId: number,
  images: GeneratedImageContract[],
  videos: readonly GeneratedVideoContract[],
  defaults: ScopedDefaults
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
      updateGalleryWithResultImages(nextProject, images, defaults),
      queueItemId,
      videos,
      defaults,
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
  videos: readonly GeneratedVideoContract[],
  defaults: ScopedDefaults
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
    resultImages: item.retriedBackendItemIds ? mergeImageResults(item.resultImages, images) : images,
    status: 'completed',
  }));

  if (destination === 'gallery') {
    return routeGalleryResultVideos(
      updateGalleryWithResultImages(nextProject, images, defaults),
      queueItemId,
      videos,
      defaults,
      nextProject
    );
  }

  // Stage only into the submitted documentRevision; a wholesale canvas swap invalidates staging.
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

const enqueueCompiledSnapshot = (
  project: Project,
  defaults: ScopedDefaults,
  route: InvocationRoute,
  runtime: Pick<InvocationRuntimeInputs, 'boardDestinationIds'>,
  compiled: Partial<ExpandedPositivePrompts> & {
    generate?: QueueGenerateSnapshot;
    graph: GraphContract;
    widgetStates: WidgetStateMap;
    workflowJson?: Record<string, unknown>;
    workflow?: Omit<WorkflowSubmissionPlan, 'graph'>;
    /** The project workflow the graph was compiled from; results and seed advances return to it. */
    projectWorkflowId?: string;
  },
  backendSupportsCancellation: boolean,
  canvasSnapshot?: CanvasStateContractV3,
  prepend = false,
  keepInvocationRoute = false
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
  const upscalePromptDraft = upscaleSettings ? getPromptDraftFromValues(widgetStates.generate?.values ?? {}) : null;
  const sourceGenerateSettings =
    route.sourceId === 'canvas'
      ? canvasGenerateSettings
      : route.sourceId === 'generate'
        ? generateSettings
        : (videoSettings ??
          (upscaleSettings && upscalePromptDraft
            ? getUpscaleSubmissionSource(upscaleSettings, upscalePromptDraft)
            : null));
  const submissionPlan = sourceGenerateSettings
    ? planGenerateSubmission(sourceGenerateSettings, compiled.positivePrompts)
    : null;
  const effectivePrompts = submissionPlan?.effectivePrompts ?? null;
  const expandedPositivePrompts = submissionPlan?.submission.positivePrompts;
  // Generate expands before dispatch, so the project's sample seed may have rotated since; follow the expansion.
  const expansionSampleSeed = expandedPositivePrompts ? (compiled.positivePromptsSampleSeed ?? null) : null;
  // Compilation already reserved the starting seed; failures and cancellations do not roll the sequence back.
  const seedPlan = submissionPlan?.seedPlan ?? null;
  const libraryWorkflowSource = compiled.projectWorkflowId
    ? project.workflows.entries.find((entry) => entry.document.id === compiled.projectWorkflowId)?.source
    : undefined;
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
            ...(libraryWorkflowSource
              ? {
                  libraryWorkflowSource: {
                    libraryWorkflowId: libraryWorkflowSource.libraryWorkflowId,
                    ...(libraryWorkflowSource.ownerUserId ? { ownerUserId: libraryWorkflowSource.ownerUserId } : {}),
                  },
                }
              : {}),
            ...(compiled.workflowJson ? { workflow: compiled.workflowJson } : {}),
            ...(compiled.projectWorkflowId ? { projectWorkflowId: compiled.projectWorkflowId } : {}),
          }
      : submissionPlan
        ? {
            ...submissionPlan.submission,
            graph: backendGraph,
            kind: 'generate',
            negativePromptNodeId: generate?.negativePromptNodeId ?? 'negative_prompt',
            positivePromptNodeId: generate?.positivePromptNodeId ?? 'positive_prompt',
            seedNodeId: generate?.seedNodeId ?? 'seed',
          }
        : { error: `${route.sourceId} queue item is missing source submission metadata.`, kind: 'invalid' };
  // A workflow compiles every node's board in (Auto included), so its board-less results stay Uncategorized.
  const galleryBoardId =
    route.sourceId === 'workflow'
      ? null
      : getGalleryAutoAddBoardId(
          applyWidgetDefaults(widgetStates.gallery?.values ?? {}, 'gallery', defaults),
          (boardId: string) => runtime.boardDestinationIds.includes(boardId)
        );
  const generatePresentationSettings = normalizeGenerateSettings(widgetStates.generate?.values);
  const videoPresentationDimensions =
    route.sourceId === 'video' && videoSettings?.model ? getVideoDimensions(videoSettings.model, videoSettings) : null;
  const presentationDimensions =
    route.sourceId === 'upscale' && upscaleSettings?.inputImage
      ? getUpscaleOutputDimensions(upscaleSettings.inputImage, upscaleSettings)
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
  const recordedCanvas =
    route.sourceId === 'canvas'
      ? Object.values(backendGraph?.nodes ?? {}).find((node) => node.type === 'core_metadata')?.canvas_v3_metadata
      : undefined;
  const recalledSampleSeed = expansionSampleSeed === null ? {} : { dynamicPromptsSampleSeed: expansionSampleSeed };
  const recall = generateRecallValues
    ? {
        ...(recordedCanvas && typeof recordedCanvas === 'object' && !Array.isArray(recordedCanvas)
          ? { canvasMetadata: { canvas_v3_metadata: structuredClone(recordedCanvas) } }
          : {}),
        generateValues: {
          ...cloneGenerateWidgetValues(generateRecallValues),
          ...recalledSampleSeed,
        },
      }
    : videoSettings
      ? { videoValues: { ...cloneVideoWidgetValues(videoSettings), ...recalledSampleSeed } }
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
        ...(effectivePrompts?.positivePrompt ? { positivePrompt: effectivePrompts.positivePrompt } : {}),
        width: presentationDimensions.width,
      },
      sourceId: route.sourceId,
      ...getQueueSourceResultSelection(route.sourceId),
      submittedAt,
      ...(recall ? { recall } : {}),
      ...(prepend ? { prepend: true } : {}),
    },
    status: 'pending',
  };

  // Advance seeds atomically with enqueueing, keeping settings edited since async compilation.
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
  const dynamicPromptsSettings = canvasGenerateSettings ?? generateSettings ?? videoSettings ?? upscalePromptDraft;
  // Fencing on the expansion's seed keeps a submission that drew an already-rotated sample from rotating again.
  const resampledProject =
    expansionSampleSeed !== null &&
    dynamicPromptsSettings &&
    dynamicPromptsSettings.dynamicPromptsCombinatorial === false &&
    dynamicPromptsSettings.dynamicPromptsResample
      ? updateProjectWidgetValues(
          advancedProject,
          route.sourceId === 'canvas' || route.sourceId === 'upscale' ? 'generate' : route.sourceId,
          (values) =>
            // Sanitized as the expansion read it: a record saved without the field drew sample seed 0.
            sanitizeSampleSeed(values.dynamicPromptsSampleSeed) === expansionSampleSeed &&
            values.dynamicPromptsCombinatorial === false
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
    invocation: keepInvocationRoute
      ? { ...project.invocation, lastSubmittedRunId: queueItemId }
      : {
          ...project.invocation,
          destination: route.destination,
          lastSubmittedRunId: queueItemId,
          sourceId: route.sourceId,
        },
    queue: { items: [queueItem, ...project.queue.items] },
    widgetGraphs:
      !keepInvocationRoute &&
      (route.sourceId === 'generate' || route.sourceId === 'upscale' || route.sourceId === 'video')
        ? { ...project.widgetGraphs, [route.sourceId]: cloneGraph(graph) }
        : project.widgetGraphs,
  };
};

const hasUpscaleInputChanged = (project: Project, metadata: UpscaleSourceMetadata | undefined): boolean => {
  const values = normalizeUpscaleWidgetValues(getWidgetValues(project, 'upscale'));

  return values !== null && !isUpscaleSourceMetadataCurrent(values, metadata?.imageName);
};

const submitModelOnlyUpscale = (
  project: Project,
  defaults: ScopedDefaults,
  {
    backendSupportsCancellation,
    runtime,
    image,
    scale,
    sourceMetadata,
    upscaleModel,
  }: Extract<WorkbenchReducerAction, { type: 'submitModelOnlyUpscale' }>
): Project => {
  const route: InvocationRoute = { ...project.invocation, destination: 'gallery', sourceId: 'upscale' };
  const compiled = compileModelOnlyUpscaleGraph({
    destination: route.destination,
    image,
    scale,
    sourceMetadata,
    upscaleModel,
  });
  const widgetStates = getWidgetStatesSnapshot(project.widgetInstances);
  const values: UpscaleWidgetValues = {
    ...createDefaultUpscaleWidgetValues(),
    inputImage: image,
    mode: 'model-only',
    scale: scale === 'native' ? (getSpandrelNativeScale(upscaleModel) ?? 1) : scale,
    seedMode: 'fixed',
    upscaleModel,
  };

  widgetStates.upscale = { ...getWidgetState(project, 'upscale'), graphId: compiled.graph.id, values: { ...values } };

  return enqueueCompiledSnapshot(
    project,
    defaults,
    route,
    runtime,
    { graph: compiled.graph, widgetStates },
    backendSupportsCancellation,
    undefined,
    false,
    true
  );
};

const submitInvocationSnapshot = (
  project: Project,
  defaults: ScopedDefaults,
  backendSupportsCancellation: boolean,
  runtime: InvocationRuntimeInputs,
  route: ResolvedInvocationRoute,
  models?: readonly ModelConfig[],
  expansion: Partial<ExpandedPositivePrompts> = {},
  workflowGenerators?: WorkflowGeneratorResolutions,
  workflowDocument?: ProjectGraphState,
  prepend = false,
  upscaleSourceMetadata?: UpscaleSourceMetadata
): Project => {
  if (!isInvocationRouteValid(route)) {
    return project;
  }

  const compiledSnapshot = compileInvocationSnapshot(
    project,
    defaults,
    route,
    runtime,
    models,
    workflowGenerators,
    workflowDocument,
    upscaleSourceMetadata
  );

  if (!compiledSnapshot) {
    return project;
  }

  return enqueueCompiledSnapshot(
    project,
    defaults,
    route,
    runtime,
    { ...compiledSnapshot, ...expansion },
    backendSupportsCancellation,
    undefined,
    prepend
  );
};

export const createInitialWorkbenchState = (): WorkbenchState => {
  const draft = createDraftProject([]);

  return {
    // A new account has no stored arrangements to adopt.
    account: { activeLayoutPresetId: defaultLayoutPreset.id, workspaceLayoutsAdopted: true },
    activeProjectId: draft.id,
    autosave: { status: 'idle' },
    enteredProjectIds: [draft.id],
    backendConnection: { status: 'connecting' },
    notifications: [],
    projects: [draft],
    // The store seeds the user's defaults (`createWorkbenchStore`); the reducer never reads the settings runtime.
    scopedDefaults: {},
    widgetFailures: [],
  };
};

const hasGallerySelectionPage = (selectionPage: number | undefined): selectionPage is number =>
  typeof selectionPage === 'number' && Number.isFinite(selectionPage);

const getGallerySelectionPage = (values: Record<string, unknown>, selectionPage: number | undefined): number =>
  hasGallerySelectionPage(selectionPage)
    ? Math.max(0, Math.floor(selectionPage))
    : typeof values.galleryPage === 'number' && Number.isFinite(values.galleryPage)
      ? Math.max(0, Math.floor(values.galleryPage))
      : 0;

const getRankingStampFields = (
  values: Record<string, unknown>,
  existingNavigationQuery: Record<string, unknown>,
  semanticKey: string
): Record<string, unknown> => {
  if (existingNavigationQuery.itemBoard === true) {
    return {};
  }

  return semanticKey
    ? {
        boardId: getGallerySelectionBoardId(values),
        mediaKind: toGalleryMediaKind(getGalleryMediaFilter(values)) ?? null,
        semanticKey,
      }
    : { semanticKey: null };
};

const getCurrentGallerySelectionQuery = (
  values: Record<string, unknown>,
  page: number,
  defaults: ScopedDefaults
): Record<string, unknown> => {
  const settings = getGallerySettings(applyWidgetDefaults(values, 'gallery', defaults));
  const semanticKey = gallerySemanticReferenceKey(parseGallerySemanticReference(values.semanticImageQuery));
  const mediaKind = toGalleryMediaKind(settings.mediaFilter);

  return {
    boardId: getGallerySelectionBoardId(values),
    galleryView: values.galleryView === 'assets' ? 'assets' : 'images',
    imageOrderDir: settings.imageOrderDir,
    ...(mediaKind ? { mediaKind } : {}),
    page,
    paginationMode: settings.paginationMode,
    searchTerm: typeof values.searchTerm === 'string' ? values.searchTerm : '',
    ...(semanticKey ? { semanticKey } : {}),
    starredOnly: values.starredOnly === true,
  };
};

/** Direct placement changes, one layout undo step each; bursts fold into one step per key. */
interface RecordedLayoutStep {
  label: string;
  mergeKey?: string;
  notice?: LayoutUndoNotice;
}

const RECORDED_LAYOUT_ACTIONS: Partial<
  Record<WorkbenchReducerAction['type'], (action: WorkbenchReducerAction) => RecordedLayoutStep>
> = {
  closeFloatingWidget: () => ({ label: 'Remove widget' }),
  dockFloatingWidget: () => ({ label: 'Dock window' }),
  floatWidget: () => ({ label: 'Float widget' }),
  keepGuest: () => ({ label: 'Add widget' }),
  moveWidgetInstance: () => ({ label: 'Move widget' }),
  openRegionWidget: () => ({ label: 'Open widget' }),
  reorderWidgetInstances: () => ({ label: 'Reorder widgets' }),
  resetCustomizeToDefault: () => ({ label: 'Reset to default' }),
  resetWorkspaceView: (action) => ({
    label: 'Reset view',
    notice: { kind: 'resetView', presetId: (action as { workspaceId: LayoutPresetId }).workspaceId },
  }),
  // Typing a name is one step per burst; an icon or route pick is its own.
  setCustomizeDraft: (action) => ({
    label: 'Edit workspace',
    mergeKey: `draft:${Object.keys((action as { draft: CustomizeDraft }).draft).join(',')}`,
  }),
  setFloatingWidgetGeometry: (action) => ({
    label: 'Move window',
    mergeKey: `geometry:${(action as { instanceId: string }).instanceId}`,
  }),
  setFloatingWidgetMode: () => ({ label: 'Change window' }),
  setRegionWidgetCollapsed: () => ({ label: 'Collapse panel' }),
  setRegionWidgetSize: (action) => ({
    label: 'Resize panel',
    mergeKey: `size:${(action as { region: string }).region}`,
  }),
  setWidgetInstanceAlignment: () => ({ label: 'Move widget' }),
  toggleRegionWidget: () => ({ label: 'Add widget' }),
};

const isSameLayoutState = (left: Project, right: Project): boolean =>
  left.workspaceId === right.workspaceId && left.views === right.views && left.invocation === right.invocation;

/**
 * Records the step a layout action made: into the project's history, with what the account held for its workspace
 * before and after when the gesture changed it there, or into the Customize session's own history while the project
 * customizes.
 */
const recordLayoutSteps = (
  previous: WorkbenchState,
  next: WorkbenchState,
  actingProjectId: string,
  step: RecordedLayoutStep
): WorkbenchState => {
  let customizing = next.customizing;
  let didChange = false;
  const projects = next.projects.map((project) => {
    const before = previous.projects.find((candidate) => candidate.id === project.id);

    if (!before) {
      return project;
    }

    const session = getCustomizeSession(next.customizing, project);
    const beforeSession = getCustomizeSession(previous.customizing, before);

    if (session && beforeSession) {
      const viewBefore = getProjectView(previous, before);
      // Only layout actions are recorded, so what moved the route here was their auto-routing, never a pick by hand.
      const route = getRouteMove(before.invocation, project.invocation);

      if (isCustomizeStep(beforeSession, session, viewBefore, getProjectView(next, project), route)) {
        customizing = {
          ...customizing,
          [project.id]: pushCustomizeHistory(session, beforeSession, viewBefore, route, step.mergeKey, now()),
        };
      }

      return project;
    }

    // A case that recorded its own step (with its notice) is left alone.
    if (before.undoRedo !== project.undoRedo) {
      return project;
    }

    const workspace =
      project.id === actingProjectId && before.workspaceId === project.workspaceId
        ? toUndoWorkspace(previous.account, next.account, project.workspaceId)
        : undefined;

    if (!workspace && isSameLayoutState(before, project)) {
      return project;
    }

    didChange = true;

    return {
      ...project,
      undoRedo: pushUndo(before, step.label, {
        mergeKey: step.mergeKey,
        ...(step.notice ? { notice: step.notice } : {}),
        ...(workspace ? { workspace } : {}),
      }).undoRedo,
    };
  });

  return didChange || customizing !== next.customizing ? { ...next, customizing, projects } : next;
};

const CUSTOMIZE_ENDED_TITLE_KEY = 'shell.workspace.customizeEndedTitle';
const CUSTOMIZE_ENDED_DELETED_MESSAGE_KEY = 'shell.workspace.customizeEndedDeletedMessage';
const CUSTOMIZE_ENDED_MOVED_MESSAGE_KEY = 'shell.workspace.customizeEndedMovedMessage';

/** Customize ends with its workspace: when the project closes or leaves it, or the workspace is deleted. */
const reconcileCustomizeSessions = (previous: WorkbenchState, next: WorkbenchState): WorkbenchState => {
  if (
    !next.customizing ||
    (next.customizing === previous.customizing &&
      next.projects === previous.projects &&
      next.account === previous.account)
  ) {
    return next;
  }

  const customizing = pruneCustomizeSessions(next.customizing, next.projects, (presetId) =>
    doesLayoutPresetExist(next.account, presetId)
  );

  if (customizing === next.customizing) {
    return next;
  }

  // A new workspace's draft ended without Done leaves no view of it behind.
  const projects = next.projects.map((project) => {
    const draftId = next.customizing?.[project.id]?.newWorkspaceId;
    const views =
      draftId === undefined || customizing?.[project.id] ? project.views : withView(project.views, draftId, undefined);

    return views === project.views ? project : { ...project, views };
  });

  const ended: WorkbenchState = {
    ...next,
    customizing,
    projects: projects.every((project, index) => project === next.projects[index]) ? next.projects : projects,
  };

  // A session with something to lose that ends while its project stays open says so, rather than vanish.
  return Object.entries(next.customizing).reduce<WorkbenchState>((announced, [projectId, session]) => {
    if (
      customizing?.[projectId] ||
      !next.projects.some(({ id }) => id === projectId) ||
      !hasCustomizeChanges(session)
    ) {
      return announced;
    }

    const saved = resolveSavedLayoutPreset(previous.account, session.presetId);
    const workspace = saved.labelKey === undefined ? { text: saved.label } : { key: saved.labelKey, text: saved.label };
    const wasDeleted = !doesLayoutPresetExist(next.account, session.presetId);

    return addNotification(
      announced,
      createNotification({
        kind: 'info',
        message: wasDeleted
          ? `${saved.label} was deleted, so the changes you were customizing were discarded.`
          : `This project moved off ${saved.label}, so the changes you were customizing were discarded.`,
        messageKey: wasDeleted ? CUSTOMIZE_ENDED_DELETED_MESSAGE_KEY : CUSTOMIZE_ENDED_MOVED_MESSAGE_KEY,
        messageListValues: { workspace: [workspace] },
        projectId,
        title: 'Customize ended',
        titleKey: CUSTOMIZE_ENDED_TITLE_KEY,
      })
    );
  }, ended);
};

/**
 * Where the active project meets a structure it may not have shown before (activation, a switch, a structural edit),
 * every instance it places gets values. Elsewhere a missing record is tolerated until then.
 */
const ensureActiveLayoutInstances = (
  previous: WorkbenchState,
  next: WorkbenchState,
  context: WorkbenchReducerContext
): WorkbenchState => {
  const project = next.projects.find((candidate) => candidate.id === next.activeProjectId);
  const before = previous.projects.find((candidate) => candidate.id === previous.activeProjectId);

  if (
    !project ||
    (before?.id === project.id &&
      before.workspaceId === project.workspaceId &&
      getProjectStructure(previous, before) === getProjectStructure(next, project))
  ) {
    return next;
  }

  const ensured = ensureLayoutInstances(project, getLayout(next, project, context));

  return ensured === project ? next : replaceProject(next, ensured);
};

export const __workbenchReducerInternal = (
  state: WorkbenchState,
  action: WorkbenchReducerAction,
  context: WorkbenchReducerContext
): WorkbenchState => {
  const reduced = reconcileCustomizeSessions(
    state,
    reconcileCanvasStagingFollow(state, reduceWorkbenchAction(state, action, context))
  );
  const entered =
    reduced.activeProjectId === state.activeProjectId && reduced.enteredProjectIds === state.enteredProjectIds
      ? reduced
      : enterArrivingProject(reduced, context);
  const describeStep = RECORDED_LAYOUT_ACTIONS[action.type];
  const recorded =
    !describeStep || entered === state
      ? entered
      : recordLayoutSteps(
          state,
          entered,
          ('projectId' in action && typeof action.projectId === 'string' ? action.projectId : undefined) ??
            state.activeProjectId,
          describeStep(action)
        );

  return recorded === state
    ? state
    : announceDroppedArrangements(ensureActiveLayoutInstances(state, recorded, context));
};

const reduceWorkbenchAction = (
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

      const settled = settleArriving(state, normalizeWorkbenchProject(action.project), context);

      return {
        ...settled.state,
        activeProjectId: settled.project.id,
        projects: [...settled.state.projects, settled.project],
      };
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
      // Opening it again later is an entry.
      const enteredProjectIds = state.enteredProjectIds?.filter((id) => id !== action.projectId);

      if (action.projectId !== state.activeProjectId) {
        return { ...state, enteredProjectIds, projects };
      }

      const fallbackProject = projects[Math.max(0, projectIndex - 1)];

      return { ...state, activeProjectId: fallbackProject.id, enteredProjectIds, projects };
    }
    case 'switchProject': {
      return { ...state, activeProjectId: action.projectId };
    }
    case 'applyPreset': {
      const preset = getAvailableLayoutPreset(state, action.presetId);
      const activeProject = state.projects.find((project) => project.id === state.activeProjectId);
      // Choosing the preset already in use goes home: navigation, not a step.
      const nextState =
        activeProject && activeProject.workspaceId !== preset.id
          ? switchActiveProjectLayoutPreset(state, preset, context)
          : enterActiveWorkspace(state, context);

      return {
        ...nextState,
        account: { ...nextState.account, activeLayoutPresetId: preset.id },
      };
    }
    case 'reorderLayoutPresets': {
      const layoutPresetOrder = reorderLayoutPresetIds(state.account, action.activeId, action.overId);

      return layoutPresetOrder ? { ...state, account: { ...state.account, layoutPresetOrder } } : state;
    }
    case 'restoreLayoutPresetDefault': {
      const account = restoreWorkspaceDefault(state.account, action.presetId);

      return isSameWorkspaceRecord(
        readWorkspaceRecord(account, action.presetId),
        readWorkspaceRecord(state.account, action.presetId)
      )
        ? state
        : { ...state, account };
    }
    case 'deleteLayoutPreset': {
      if (isBuiltInLayoutPresetId(action.presetId) || !doesLayoutPresetExist(state.account, action.presetId)) {
        return state;
      }

      const layoutPresetOrder = getOrderedLayoutPresets(state.account)
        .map(({ id }) => id)
        .filter((id) => id !== action.presetId);
      const customLayoutPresets = (state.account.customLayoutPresets ?? []).filter(
        (preset) => preset.id !== action.presetId
      );
      const account: AccountState = {
        ...state.account,
        activeLayoutPresetId:
          state.account.activeLayoutPresetId === action.presetId
            ? defaultLayoutPreset.id
            : state.account.activeLayoutPresetId,
        customLayoutPresets,
        layoutPresetOrder,
        ...(state.account.writtenStructureIds?.includes(action.presetId)
          ? { writtenStructureIds: state.account.writtenStructureIds.filter((id) => id !== action.presetId) }
          : {}),
      };
      const source: LayoutSource = { account, customizing: state.customizing };
      // Projects on it move to the default workspace as a switch would; every project forgets its view of it.
      const projects = state.projects.map((project) => {
        const views = withView(project.views, action.presetId, undefined);

        return project.workspaceId === action.presetId
          ? enterProjectWorkspace(source, { ...project, views, workspaceId: defaultLayoutPreset.id }, context)
          : views === project.views
            ? project
            : { ...project, views };
      });

      return { ...state, account, projects };
    }
    case 'beginCustomize': {
      const activeProject = state.projects.find((project) => project.id === state.activeProjectId);
      const { newWorkspace } = action;

      if (
        !activeProject ||
        getCustomizeSession(state.customizing, activeProject) ||
        (newWorkspace && !isNewWorkspaceId(state.account, newWorkspace.id))
      ) {
        return state;
      }

      const entryView = getProjectView(state, activeProject);
      const saved = getWorkspaceStructure(state.account, activeProject.workspaceId);
      // A new workspace starts from what is on screen, seen through an empty view of its own.
      const structure = newWorkspace
        ? captureLayout(saved, entryView, getLayout(state, activeProject, context), context.availability)
        : saved;

      return {
        ...updateActiveProject(state, withoutLayoutHistory),
        customizing: {
          ...state.customizing,
          [activeProject.id]: beginCustomizeSession({
            entryView,
            ...(newWorkspace ? { newWorkspace } : {}),
            presetId: activeProject.workspaceId,
            structure,
          }),
        },
      };
    }
    case 'resetCustomizeToDefault': {
      const activeProject = state.projects.find((project) => project.id === state.activeProjectId);
      const session = activeProject && getCustomizeSession(state.customizing, activeProject);
      const reset = session && resetCustomizeSession(session);

      // The account keeps its overrides until Done; Undo withdraws the reset with the draft and view it replaced.
      return reset
        ? replaceProject(
            { ...state, customizing: { ...state.customizing, [activeProject.id]: reset } },
            { ...activeProject, views: withView(activeProject.views, activeProject.workspaceId, undefined) }
          )
        : state;
    }
    case 'setCustomizeDraft': {
      const activeProject = state.projects.find((project) => project.id === state.activeProjectId);
      const session = activeProject && getCustomizeSession(state.customizing, activeProject);
      const edited = session && editCustomizeDraft(session, action.draft);

      return edited && edited !== session
        ? { ...state, customizing: { ...state.customizing, [activeProject.id]: edited } }
        : state;
    }
    case 'saveCustomizeAsNew': {
      const activeProject = state.projects.find((project) => project.id === state.activeProjectId);
      const session = activeProject && getCustomizeSession(state.customizing, activeProject);
      const saving =
        session && isNewWorkspaceId(state.account, action.workspace.id)
          ? saveCustomizeSessionAsNew(session, action.workspace)
          : null;

      if (!activeProject || !session || !saving) {
        return state;
      }

      // The draft's view goes with the copy; the original gets back the view it had before Customize.
      return replaceProject(
        { ...state, customizing: { ...state.customizing, [activeProject.id]: saving } },
        {
          ...activeProject,
          views: withView(
            withView(activeProject.views, activeProject.workspaceId, session.entryView),
            action.workspace.id,
            getProjectView(state, activeProject)
          ),
        }
      );
    }
    case 'resetWorkspaceView': {
      return updateActiveProject(state, (project) => {
        const views = withView(project.views, action.workspaceId, undefined);

        return views === project.views ? project : { ...project, views };
      });
    }
    case 'releaseHeldLayouts': {
      // Only a session write that carried the adoption saved it; one sent before the adoption did not.
      if (!state.layoutAdoptionPending || action.savedAccount.workspaceLayoutsAdopted !== true) {
        return state;
      }

      const { layoutAdoptionPending: _pending, ...released } = state;

      return { ...released, projects: state.projects.map(releaseHeldLayout) };
    }
    case 'clearLayoutHistory': {
      const projects = state.projects.map((project) =>
        getCustomizeSession(state.customizing, project) ? project : withoutLayoutHistory(project)
      );

      return projects.every((project, index) => project === state.projects[index]) ? state : { ...state, projects };
    }
    case 'cancelCustomize': {
      const activeProject = state.projects.find((project) => project.id === state.activeProjectId);
      const session = activeProject && getCustomizeSession(state.customizing, activeProject);

      if (!activeProject || !session) {
        return state;
      }

      const { [activeProject.id]: _ended, ...customizing } = state.customizing ?? {};
      const views = withView(activeProject.views, activeProject.workspaceId, session.entryView);

      // The route goes back only as far as Customize's auto-routing moved it; a pick made by hand stays.
      return replaceProject(
        { ...state, customizing },
        withoutLayoutHistory({
          ...activeProject,
          invocation:
            stepRouteMove(activeProject.invocation, getCustomizeRouteReturn(session, activeProject.invocation)) ??
            activeProject.invocation,
          views: session.newWorkspaceId === undefined ? views : withView(views, session.newWorkspaceId, undefined),
        })
      );
    }
    case 'finishCustomize': {
      const activeProject = state.projects.find((project) => project.id === state.activeProjectId);
      const session = activeProject && getCustomizeSession(state.customizing, activeProject);

      if (!activeProject || !session) {
        return state;
      }

      const view = getProjectView(state, activeProject);
      const captured = captureLayout(
        session.structure,
        view,
        getLayout(state, activeProject, context),
        context.availability
      );
      const account = finishCustomizeSession(state.account, session, captured, view !== session.entryView);

      if (!account) {
        return state;
      }

      const { [activeProject.id]: _ended, ...customizing } = state.customizing ?? {};
      const workspaceId = session.newWorkspaceId ?? session.presetId;
      const saved = getWorkspaceStructure(account, workspaceId);
      // What was on screen is the workspace's defaults now; the view keeps only what they cannot say.
      const views =
        session.newWorkspaceId === undefined && saved === getWorkspaceStructure(state.account, workspaceId)
          ? withView(activeProject.views, workspaceId, view)
          : withView(activeProject.views, workspaceId, pruneView(view, saved, context.availability));
      const finished = withoutLayoutHistory({ ...activeProject, views, workspaceId });
      // Done on an existing workspace is one step back to before Customize, while nothing has changed it since.
      const workspace =
        session.newWorkspaceId === undefined ? toUndoWorkspace(state.account, account, workspaceId) : undefined;
      const beforeCustomize = { ...finished, views: withView(views, workspaceId, session.entryView) };
      const routeReturn = getCustomizeRouteReturn(session, finished.invocation);

      return replaceProject(
        { ...state, account, customizing },
        workspace
          ? {
              ...finished,
              undoRedo: pushUndo(beforeCustomize, 'Update workspace', {
                customizeRoute: routeReturn ? { move: routeReturn } : {},
                notice: { kind: 'updateWorkspace', presetId: workspaceId },
                workspace,
              }).undoRedo,
            }
          : finished
      );
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
      const project = state.projects.find((candidate) => candidate.id === (action.projectId ?? state.activeProjectId));

      if (!project) {
        return state;
      }

      const structureAction: Extract<StructureAction, { type: 'addWidget' }> = {
        ...(action.createNew ? { createNew: true } : {}),
        mintId: createId(`widget-${action.widgetId}`),
        region: action.region,
        type: 'addWidget',
        typeId: action.widgetId,
        // An instance of the type this project already has keeps its values when it is placed again.
        unplacedInstanceId: Object.values(project.widgetInstances).find(
          (instance) => instance.typeId === action.widgetId
        )?.id,
      };
      const instanceId = getAddedInstanceId(getProjectStructure(state, project), structureAction);
      const withValues = project.widgetInstances[instanceId]
        ? project
        : {
            ...project,
            widgetInstances: {
              ...project.widgetInstances,
              [instanceId]: createWidgetInstance(action.widgetId, instanceId, action.initialValues),
            },
          };

      const compose = (next: Project, source: LayoutSource): Project =>
        applyAutoRouteForWidgetReveal(
          showInRegion(source, next, action.region, instanceId, context.availability),
          action.widgetId,
          context
        );

      // One the region holds already is only fronted.
      return getProjectStructure(state, project).widgetRegions[action.region].instanceIds.includes(instanceId)
        ? replaceProject(state, compose(project, state))
        : applyStructureGesture(state, withValues, structureAction, context, { compose });
    }
    case 'selectRegionWidget': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) => {
        const region = getLayout(state, project, context).widgetRegions[action.region];

        // A stale request for an instance the region no longer holds changes nothing; it must not discard the window.
        if (!region.instanceIds.includes(action.widgetId)) {
          return project;
        }

        if (action.region === 'center' || region.activeInstanceId !== action.widgetId) {
          return applyAutoRouteForRevealedInstance(
            showInRegion(state, project, action.region, action.widgetId, context.availability),
            action.widgetId,
            context
          );
        }

        // The front tab of a side region discloses or collapses it.
        const disclosed = updateProjectView(state, project, (view, structure) =>
          setRegionCollapsed(view, structure, action.region, !region.isCollapsed)
        );

        return region.isCollapsed ? applyAutoRouteForRevealedInstance(disclosed, action.widgetId, context) : disclosed;
      });
    }
    case 'frontRegionWidget': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) => {
        const region = getLayout(state, project, context).widgetRegions[action.region];

        // View only: an instance the region does not hold is never added, and one already shown is left alone.
        if (
          !region.instanceIds.includes(action.instanceId) ||
          (region.activeInstanceId === action.instanceId && !region.isCollapsed)
        ) {
          return project;
        }

        return applyAutoRouteForRevealedInstance(
          showInRegion(state, project, action.region, action.instanceId, context.availability),
          action.instanceId,
          context
        );
      });
    }
    case 'toggleRegionWidget': {
      const project = state.projects.find((candidate) => candidate.id === (action.projectId ?? state.activeProjectId));
      const structure = project && getProjectStructure(state, project);

      // A refused toggle must keep object identity: any new `projects` reference triggers an autosave.
      if (!project || !structure || structure.floatingWidgets?.[action.widgetId]) {
        return state;
      }

      const previousRegion = getLayout(state, project, context).widgetRegions[action.region];

      if (structure.widgetRegions[action.region].instanceIds.includes(action.widgetId)) {
        return applyStructureGesture(
          state,
          project,
          { instanceId: action.widgetId, region: action.region, type: 'removeWidget' },
          context,
          {
            compose: (next, source) =>
              applyAutoRouteForRegionFront(
                next,
                previousRegion,
                getLayout(source, next, context).widgetRegions[action.region],
                context
              ),
            notice: { label: 'Remove widget', value: { instanceId: action.widgetId, kind: 'removeWidget' } },
          }
        );
      }

      const typeId =
        project.widgetInstances[action.widgetId]?.typeId ??
        structure.widgetInstances[action.widgetId]?.typeId ??
        action.widgetId.split(':', 1)[0]!;

      return applyStructureGesture(
        state,
        project,
        { createNew: true, mintId: action.widgetId, region: action.region, type: 'addWidget', typeId },
        context,
        {
          compose: (next, source) => {
            const fronted = updateProjectView(source, next, (view, edited) =>
              frontRegion(view, edited, action.region, action.widgetId, context.availability)
            );

            return applyAutoRouteForRegionFront(
              fronted,
              previousRegion,
              getLayout(source, fronted, context).widgetRegions[action.region],
              context
            );
          },
        }
      );
    }
    case 'floatWidget': {
      const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

      if (!project?.widgetInstances[action.instanceId]) {
        return state;
      }

      const remembered = project.lastFloatingGeometry?.[action.instanceId];

      return applyStructureGesture(
        state,
        project,
        {
          ...(remembered
            ? { geometry: action.viewport ? fitWindowIntoViewport(remembered, action.viewport) : remembered }
            : {}),
          instanceId: action.instanceId,
          ...(action.region ? { region: action.region } : {}),
          type: 'floatWidget',
        },
        context,
        { compose: (next) => applyAutoRouteForRevealedInstance(next, action.instanceId, context) }
      );
    }
    case 'dockFloatingWidget': {
      const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);
      const window = project && getProjectStructure(state, project).floatingWidgets?.[action.instanceId];

      if (!project || !window) {
        return state;
      }

      return applyStructureGesture(state, project, { instanceId: action.instanceId, type: 'dockWindow' }, context, {
        compose: (next, source) =>
          applyAutoRouteForRevealedInstance(
            showInRegion(source, next, window.returnRegion, action.instanceId, context.availability),
            action.instanceId,
            context
          ),
      });
    }
    case 'closeFloatingWidget': {
      const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

      return project
        ? applyStructureGesture(state, project, { instanceId: action.instanceId, type: 'closeWindow' }, context, {
            notice: { label: 'Remove widget', value: { instanceId: action.instanceId, kind: 'removeWidget' } },
          })
        : state;
    }
    case 'setFloatingWidgetGeometry': {
      return updateActiveProject(state, (project) =>
        updateProjectView(state, project, (view, structure) =>
          setWindowGeometry(view, structure, action.instanceId, {
            heightPx: action.heightPx,
            widthPx: action.widthPx,
            x: action.x,
            y: action.y,
          })
        )
      );
    }
    case 'setFloatingWidgetMode': {
      return updateActiveProject(state, (project) =>
        updateProjectView(state, project, (view, structure) =>
          setWindowMode(view, structure, action.instanceId, action.mode)
        )
      );
    }
    case 'raiseFloatingWidget':
    case 'revealFloatingWidget': {
      return updateActiveProject(state, (project) => {
        const raised = updateProjectView(state, project, (view, structure) => {
          const top = raiseWindow(view, structure, action.instanceId);

          // Revealing a shaded window expands it too.
          return action.type === 'revealFloatingWidget' &&
            getWindowView(structure, top, action.instanceId)?.mode === 'shaded'
            ? setWindowMode(top, structure, action.instanceId, 'windowed')
            : top;
        });

        // Only an actual raise should re-route or dirty the project.
        return raised === project ? project : applyAutoRouteForRevealedInstance(raised, action.instanceId, context);
      });
    }
    case 'showGuest': {
      return updateActiveProject(state, (project) => {
        const structure = getProjectStructure(state, project);
        const guest = { surface: action.surface, typeId: action.typeId };
        const { getPresentation } = context.availability;
        const canShow =
          action.surface === 'center'
            ? getPresentation(action.typeId, 'center') === 'view'
            : getGuestWindowRegion(getPresentation, action.typeId) !== null;
        // The guest of the type this view shows already is the one shown again.
        const shownId = Object.entries(getProjectView(state, project)?.guests ?? {}).find(
          ([id, shown]) => shown.typeId === action.typeId && !Object.hasOwn(structure.widgetInstances, id)
        )?.[0];
        const instanceId =
          shownId ??
          getGuestInstanceId(
            structure,
            action.typeId,
            Object.values(project.widgetInstances)
              .filter((instance) => instance.typeId === action.typeId)
              .map((instance) => instance.id)
          );
        const existing = project.widgetInstances[instanceId];

        if (
          !canShow ||
          isGuestSuperseded(structure, instanceId, guest, context.availability) ||
          (existing && existing.typeId !== action.typeId)
        ) {
          return project;
        }

        const windowCount = Object.keys(getLayout(state, project, context).floatingWidgets ?? {}).length;
        const remembered = project.lastFloatingGeometry?.[instanceId] ?? cascadeDefaultGeometry(windowCount);
        const geometry = action.viewport ? fitWindowIntoViewport(remembered, action.viewport) : remembered;
        const withValues = existing
          ? project
          : {
              ...project,
              widgetInstances: {
                ...project.widgetInstances,
                [instanceId]: createWidgetInstance(action.typeId, instanceId, action.initialValues),
              },
            };
        const shown = updateProjectView(state, withValues, (view, edited) =>
          showGuest(view, edited, instanceId, guest, geometry)
        );

        return shown === withValues ? project : applyAutoRouteForRevealedInstance(shown, instanceId, context);
      });
    }
    case 'closeGuest': {
      return updateActiveProject(state, (project) => {
        const layout = getLayout(state, project, context);

        // Close reads only `canClose`, which no multiplicity changes.
        if (!getShownGuest(layout, action.instanceId, context.availability, false)?.canClose) {
          return project;
        }

        const closed = updateProjectView(state, project, (view) => removeGuest(view, action.instanceId));

        return closed === project ? project : rememberLayoutWindows(closed, layout, getLayout(state, closed, context));
      });
    }
    case 'keepGuest': {
      const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);
      const layout = project && getLayout(state, project, context);
      // The resolved layout's guests include the center's fallback stage, which no view records.
      const guest = layout && getShownGuest(layout, action.instanceId, context.availability, action.allowMultiple);
      const window = layout?.floatingWidgets?.[action.instanceId];

      if (!project || !guest?.canKeep) {
        return state;
      }

      if (guest.surface === 'center') {
        // As every add: a type allowing one instance lands on the one the workspace may place already.
        const add: Extract<StructureAction, { type: 'addWidget' }> = {
          ...(action.allowMultiple ? { createNew: true } : {}),
          mintId: action.instanceId,
          region: 'center',
          type: 'addWidget',
          typeId: guest.typeId,
        };
        const kept = getAddedInstanceId(getProjectStructure(state, project), add);

        return applyStructureGesture(state, project, add, context, {
          compose: (next, source) => showInRegion(source, next, 'center', kept, context.availability),
        });
      }

      return window
        ? applyStructureGesture(
            state,
            project,
            {
              geometry: { heightPx: window.heightPx, widthPx: window.widthPx, x: window.x, y: window.y },
              instanceId: action.instanceId,
              region: window.returnRegion,
              type: 'addWindow',
              typeId: guest.typeId,
            },
            context
          )
        : state;
    }
    case 'moveWidgetInstance': {
      const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

      return project
        ? applyStructureGesture(
            state,
            project,
            {
              fromRegion: action.fromRegion,
              instanceId: action.instanceId,
              toIndex: action.toIndex,
              toRegion: action.toRegion,
              type: 'moveWidget',
            },
            context,
            {
              compose: (next, source) =>
                applyAutoRouteForRevealedInstance(
                  showInRegion(source, next, action.toRegion, action.instanceId, context.availability),
                  action.instanceId,
                  context
                ),
            }
          )
        : state;
    }
    case 'reorderWidgetInstances': {
      const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

      if (!project) {
        return state;
      }

      const previousRegion = getLayout(state, project, context).widgetRegions[action.region];
      const front = (next: Project, source: LayoutSource): Project => {
        const fronted =
          action.activeInstanceId === undefined
            ? next
            : updateProjectView(source, next, (view, structure) =>
                frontRegion(view, structure, action.region, action.activeInstanceId!, context.availability)
              );

        return applyAutoRouteForRegionFront(
          fronted,
          previousRegion,
          getLayout(source, fronted, context).widgetRegions[action.region],
          context
        );
      };
      const reordered = applyStructureGesture(
        state,
        project,
        { instanceIds: action.instanceIds, region: action.region, type: 'reorderWidgets' },
        context,
        { compose: front }
      );

      // A drop in place still fronts what was dropped.
      return reordered !== state ? reordered : replaceProject(state, front(project, state));
    }
    case 'setWidgetInstanceAlignment': {
      const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

      return project
        ? applyStructureGesture(
            state,
            project,
            { align: action.align, instanceId: action.instanceId, region: action.region, type: 'setAlignment' },
            context
          )
        : state;
    }
    case 'setRegionWidgetCollapsed': {
      return updateActiveProject(state, (project) =>
        updateProjectView(state, project, (view, structure) =>
          setRegionCollapsed(view, structure, action.region, action.isCollapsed)
        )
      );
    }
    case 'setRegionWidgetSize': {
      return updateActiveProject(state, (project) =>
        updateProjectView(state, project, (view, structure) =>
          setRegionSize(view, structure, action.region, action.sizePx)
        )
      );
    }
    case 'setGenerateSettings': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) => {
        const updated = updateProjectWidgetValues(project, 'generate', () => cloneGenerateWidgetValues(action.values));

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
    case 'restorePromptHistory': {
      return updateProjectById(state, action.projectId, (project) => ({
        ...project,
        promptHistory: restorePromptHistoryItems(project.promptHistory, action.items),
      }));
    }
    case 'patchWidgetValues': {
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
    case 'clearWidgetInstanceValues': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) =>
        updateProjectWidgetInstanceValues(project, action.instanceId, (values) => {
          if (!action.keys.some((key) => Object.hasOwn(values, key))) {
            return values;
          }
          const next = { ...values };
          for (const key of action.keys) {
            delete next[key];
          }
          return next;
        })
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
    case 'submitResolvedInvocationSnapshot': {
      const target = state.projects.find((project) => project.id === action.projectId);
      const workflow =
        target && action.route.sourceId === 'workflow'
          ? action.workflowId
            ? findProjectWorkflow(target, action.workflowId)
            : getActiveProjectWorkflow(target)
          : undefined;

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

      // The prompt or its expansion settings changed mid-expansion; neither the stale expansion nor the raw prompt runs.
      if (
        target &&
        !isSameDynamicPromptsExpansionSource(
          action.expandedFrom ?? null,
          getDynamicPromptsExpansionSource(target, action.route.sourceId)
        )
      ) {
        return addNotification(
          state,
          createNotification({
            category: 'enqueue',
            kind: 'error',
            projectId: target.id,
            title: 'The prompt changed while it was being expanded, so nothing was queued. Invoke again.',
            titleKey: 'topbar.invoke.expansionStale',
          })
        );
      }

      if (
        target &&
        action.route.sourceId === 'upscale' &&
        hasUpscaleInputChanged(target, action.upscaleSourceMetadata)
      ) {
        return addNotification(
          state,
          createNotification({
            category: 'enqueue',
            kind: 'error',
            projectId: target.id,
            title: 'The input image changed while the upscale was being prepared, so nothing was queued. Invoke again.',
            titleKey: 'widgets.upscale.inputChanged',
          })
        );
      }

      return withEnqueueNotification(
        state,
        updateProjectById(state, action.projectId, (project) =>
          submitInvocationSnapshot(
            project,
            state.scopedDefaults,
            action.backendSupportsCancellation,
            action.runtime,
            resolveInvocationRoute(
              project,
              getLayout(state, project, context),
              'global',
              action.route,
              action.models,
              workflow?.document,
              action.runtime.templates
            ),
            action.models,
            { positivePrompts: action.positivePrompts, positivePromptsSampleSeed: action.positivePromptsSampleSeed },
            action.workflowGenerators,
            workflow?.document,
            action.prepend === true,
            action.upscaleSourceMetadata
          )
        ),
        action.projectId
      );
    }
    case 'submitModelOnlyUpscale': {
      return withEnqueueNotification(
        state,
        updateProjectById(state, action.projectId, (project) =>
          submitModelOnlyUpscale(project, state.scopedDefaults, action)
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

          return item.backendBatchId === action.backendBatchId &&
            hasSameBackendItemIds &&
            item.status === status &&
            item.submissionRetry === undefined
            ? item
            : {
                ...withoutSubmissionRetry(item),
                backendBatchId: action.backendBatchId,
                backendItemIds: action.backendItemIds,
                status,
              };
        });

        return clampCanvasStagingSelection(nextProject, previousSelectedSlot);
      });
    }
    case 'recordQueueItemSubmissionRetry': {
      return updateProjectById(state, action.projectId, (project) =>
        updateQueueItem(project, action.queueItemId, (item) =>
          item.status !== 'pending' ||
          item.backendItemIds?.length ||
          isSameSubmissionRetry(item.submissionRetry, action.retry)
            ? item
            : { ...item, submissionRetry: { ...action.retry } }
        )
      );
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
          ...(action.status === 'pending' ? item : withoutSubmissionRetry(item)),
          error: action.error,
          status: action.status,
        }));

        return clampCanvasStagingSelection(nextProject, previousSelectedSlot);
      });

      if (action.notify === false || !project || (action.status !== 'failed' && action.status !== 'cancelled')) {
        return nextState;
      }

      return addNotification(
        nextState,
        createRunOutcomeNotification(project, queueItem, {
          error: action.error,
          errorCode: action.errorCode,
          failedBackendItemIds: action.failedBackendItemIds,
          status: action.status,
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
        routeQueueItemPartialResults(
          project,
          action.queueItemId,
          action.backendItemId,
          action.images,
          action.videos ?? [],
          state.scopedDefaults
        )
      );
    }
    case 'noteQueueItemFirstProgressFrame': {
      return recordCanvasStagingFrame(state, action.projectId, action.queueItemId, action.itemIndex);
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
        routeQueueItemResults(project, action.queueItemId, action.images, action.videos ?? [], state.scopedDefaults)
      );

      if (action.images.length === 0) {
        return nextState;
      }

      const count = action.images.length;

      return addNotification(
        nextState,
        createNotification({
          ...withRunLabel(
            queueItem,
            (label) => ({
              message: `${count} image${count === 1 ? '' : 's'} from “${label}”`,
              messageKey: 'notifications.run.completedImages',
              messageValues: { count, label },
            }),
            {
              message: `${count} image${count === 1 ? '' : 's'}`,
              messageKey: 'notifications.run.completedImagesUnlabeled',
              messageValues: { count },
            }
          ),
          kind: 'success',
          projectId: action.projectId,
          title: project ? `Finished in ${project.name}` : 'Run finished',
          ...(project
            ? { titleKey: 'notifications.run.completed', titleValues: { project: project.name } }
            : { titleKey: 'notifications.run.completedUnnamed' }),
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
    case 'reopenQueueItemForRetry': {
      return updateProjectById(state, action.projectId, (project) => {
        const reopened = reopenQueueItemForRetry(project, action.queueItemId, action.retriedItems);

        return reopened === project
          ? project
          : clampCanvasStagingSelection(reopened, getSelectedCanvasStagingSlot(project));
      });
    }
    case 'stageRetriedCanvasResults': {
      return updateProjectById(state, action.projectId, (project) =>
        clampCanvasStagingSelection(
          stageCanvasImages(
            project,
            action.queueItemId,
            getRetriedCanvasResultDocument(project, action.queueItemId),
            action.images,
            action.images.map(() => action.backendItemId),
            getSelectedCanvasStagingSlot(project)
          )
        )
      );
    }
    case 'appendCanvasStagingCandidate': {
      return updateProjectById(state, action.projectId, (project) =>
        appendCanvasStagingCandidate(project, action.candidate)
      );
    }
    case 'selectGalleryItem': {
      return updateGalleryValuesAndStopAutoSwitch(
        state,
        (values) => {
          const selectedImagePage =
            action.navigateItemBoard && !hasGallerySelectionPage(action.selectionPage)
              ? 0
              : getGallerySelectionPage(values, action.selectionPage);
          const semanticKey = gallerySemanticReferenceKey(parseGallerySemanticReference(values.semanticImageQuery));
          const existingNavigationQuery =
            values.selectedImageQuery && typeof values.selectedImageQuery === 'object'
              ? (values.selectedImageQuery as Record<string, unknown>)
              : null;
          const settings = getGallerySettings(applyWidgetDefaults(values, 'gallery', state.scopedDefaults));
          const selectedImageQuery = action.navigateItemBoard
            ? {
                boardId: action.item.boardId,
                galleryView: getGalleryItemView(action.item) ?? getGalleryView(values),
                imageOrderDir: settings.imageOrderDir,
                itemBoard: true,
                page: selectedImagePage,
                paginationMode: settings.paginationMode,
                searchTerm: '',
                starredOnly: false,
              }
            : action.preserveNavigationQuery && existingNavigationQuery
              ? {
                  ...existingNavigationQuery,
                  ...getRankingStampFields(values, existingNavigationQuery, semanticKey),
                  page: selectedImagePage,
                }
              : getCurrentGallerySelectionQuery(values, selectedImagePage, state.scopedDefaults);
          const itemKey = toGalleryItemKey(action.item);

          return {
            ...values,
            ...getSelectionComparePatch(values, action.item),
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
      return updateGalleryValuesAndStopAutoSwitch(
        state,
        (values) => {
          const itemKey = toGalleryItemKey(action.item);
          const selectedItemKeys = getPersistedSelectedGalleryItemKeys(values);
          const isCompareArmed = getGalleryCompareState(values) === 'armed';

          if (!selectedItemKeys.includes(itemKey)) {
            const selectedImagePage = getGallerySelectionPage(values, action.selectionPage);

            return {
              ...values,
              ...(action.item.kind === 'video' || isCompareArmed ? { compareImage: null } : {}),
              selectedImage: action.item,
              selectedImageName: itemKey,
              selectedImageNames: [...selectedItemKeys, itemKey],
              selectedImagePage,
              selectedImageQuery: getCurrentGallerySelectionQuery(values, selectedImagePage, state.scopedDefaults),
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
              ...(isCompareArmed ? { compareImage: null } : {}),
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
          const nextPrimaryPage =
            nextPrimaryItem && hasGallerySelectionPage(action.selectionPage)
              ? getGallerySelectionPage(values, action.selectionPage)
              : null;

          return {
            ...values,
            ...(nextPrimaryItem?.kind === 'image' && !isCompareArmed ? {} : { compareImage: null }),
            selectedImage: nextPrimaryItem,
            selectedImageName: nextPrimaryKey,
            selectedImageNames: expectedNextPrimaryKey && !nextPrimaryItem ? [] : remainingItemKeys,
            ...(nextPrimaryPage === null
              ? {}
              : {
                  selectedImagePage: nextPrimaryPage,
                  selectedImageQuery: getCurrentGallerySelectionQuery(values, nextPrimaryPage, state.scopedDefaults),
                }),
          };
        },
        action.projectId
      );
    }
    case 'setGalleryMultiSelection': {
      return updateGalleryValuesAndStopAutoSwitch(
        state,
        (values) => {
          const semanticKey = gallerySemanticReferenceKey(parseGallerySemanticReference(values.semanticImageQuery));
          const selectedImagePage = getGallerySelectionPage(values, action.selectionPage);
          const existingNavigationQuery =
            values.selectedImageQuery && typeof values.selectedImageQuery === 'object'
              ? (values.selectedImageQuery as Record<string, unknown>)
              : null;

          return {
            ...values,
            ...(action.primaryItem.kind === 'video' || getGalleryCompareState(values) === 'armed'
              ? { compareImage: null }
              : {}),
            selectedImage: action.primaryItem,
            selectedImageName: toGalleryItemKey(action.primaryItem),
            selectedImageNames: action.itemKeys,
            selectedImagePage,
            selectedImageQuery:
              action.preserveNavigationQuery && existingNavigationQuery
                ? {
                    ...existingNavigationQuery,
                    ...getRankingStampFields(values, existingNavigationQuery, semanticKey),
                    page: selectedImagePage,
                  }
                : getCurrentGallerySelectionQuery(values, selectedImagePage, state.scopedDefaults),
          };
        },
        action.projectId
      );
    }
    case 'setGalleryCompareImage': {
      const updateValues = (values: Record<string, unknown>) => ({ ...values, compareImage: action.image });

      return action.image
        ? updateGalleryValuesAndStopAutoSwitch(state, updateValues, action.projectId)
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
          // A semantic search follows the board switch; a map cluster ends with an actual board change.
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
        (values) => ({
          ...values,
          compareImage: null,
          selectedImage: null,
          selectedImageName: null,
          selectedImageNames: [],
        }),
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
      const resetsQuery =
        action.settings.imageOrderDir !== undefined ||
        action.settings.mediaFilter !== undefined ||
        action.settings.paginationMode !== undefined;

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
      return removeGalleryItemsFromAllProjects(
        state,
        new Set(action.itemKeys),
        action.keepComparisonForSuccessor ? (action.projectId ?? state.activeProjectId) : null
      );
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

      return reconcileDeletedGalleryBoard(state, outcome.boardId, deletedItemKeys, confirmedMovedItemKeys);
    }
    case 'setGalleryProjectBoardId': {
      return updateGalleryValues(state, (values) => ({ ...values, projectBoardId: action.boardId }), action.projectId);
    }
    case 'applyCanvasProjectMutation': {
      const nextState = updateProjectById(state, action.projectId, (project) => {
        const updated = applyCanvasProjectMutation(project, action.mutation);

        return updated !== project && action.origin !== 'system' && isHighConfidenceCanvasEdit(action.mutation)
          ? applyAutoRouteForEdit(updated, 'canvas', context)
          : updated;
      });

      return updateCanvasStagingPin(nextState, action.projectId, action.mutation);
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
            state.scopedDefaults,
            { ...project.invocation, destination: action.destination, sourceId: 'canvas' },
            action.runtime,
            {
              generate: action.generate,
              graph: action.graph,
              positivePrompts: action.positivePrompts,
              positivePromptsSampleSeed: action.positivePromptsSampleSeed,
              widgetStates: getWidgetStatesSnapshot(project.widgetInstances),
            },
            action.backendSupportsCancellation,
            action.canvas,
            action.prepend === true
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

              return { ...withoutSubmissionRetry(item), cancellationPending: true, status: 'cancelled' };
            }),
          },
        };

        return clampCanvasStagingSelection(nextProject, previousSelectedSlot);
      });

      if (!targetProject || !queueItem || !canCancelQueueItem || action.notice === null) {
        return nextState;
      }

      return addNotification(
        nextState,
        createNotification({
          kind: 'info',
          projectId: targetProject.id,
          ...(action.notice
            ? { message: action.notice.message, title: action.notice.title }
            : {
                ...describeRunLabel(queueItem),
                title: `Stopping a run in ${targetProject.name}`,
                titleKey: 'notifications.run.cancelRequested',
                titleValues: { project: targetProject.name },
              }),
        })
      );
    }
    case 'cancelUnsubmittedQueueItems': {
      if (
        !state.projects.some(
          (project) =>
            (action.projectId === undefined || project.id === action.projectId) &&
            project.queue.items.some(isUnsubmittedQueueItem)
        )
      ) {
        return state;
      }

      return {
        ...state,
        projects: state.projects.map((project) => {
          if (action.projectId !== undefined && project.id !== action.projectId) {
            return project;
          }

          const previousSelectedSlot = getSelectedCanvasStagingSlot(project);
          const nextProject: Project = {
            ...project,
            queue: {
              items: project.queue.items.map((item) =>
                isUnsubmittedQueueItem(item)
                  ? { ...withoutSubmissionRetry(item), cancellationPending: true, status: 'cancelled' }
                  : item
              ),
            },
          };

          return clampCanvasStagingSelection(nextProject, previousSelectedSlot);
        }),
      };
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
    case 'undoProjectChange':
    case 'redoProjectChange': {
      const project = state.projects.find((candidate) => candidate.id === (action.projectId ?? state.activeProjectId));
      const direction = action.type === 'undoProjectChange' ? 'undo' : 'redo';
      const session = project && getCustomizeSession(state.customizing, project);

      if (!project) {
        return state;
      }
      if (session) {
        const stepped = stepCustomizeHistory(
          session,
          getProjectView(state, project),
          project.invocation,
          direction,
          now()
        );

        return stepped
          ? replaceProject(
              { ...state, customizing: { ...state.customizing, [project.id]: stepped.session } },
              {
                ...project,
                invocation: stepped.invocation,
                views: withView(project.views, getProjectViewId(state, project), stepped.view),
              }
            )
          : state;
      }
      if (action.type === 'undoProjectChange' && action.throughEntryId !== undefined) {
        const index = project.undoRedo.past.findIndex((entry) => entry.id === action.throughEntryId);
        let restored = state;

        for (let step = index === -1 ? 0 : project.undoRedo.past.length - index; step > 0; step--) {
          restored = stepProjectHistory(restored, project.id, 'undo');
        }

        return restored;
      }

      return stepProjectHistory(state, project.id, direction);
    }
    case 'hydrateWorkbench': {
      return {
        ...normalizeWorkbenchState(action.state, context),
        backendConnection: state.backendConnection,
        enteredProjectIds: [],
        scopedDefaults: state.scopedDefaults,
      };
    }
    case 'replaceProjectFromServer': {
      const settled = settleArriving(state, normalizeWorkbenchProject(action.project), context);
      const normalizedServerProject = settled.project;
      const localProject = settled.state.projects.find((project) => project.id === action.projectId);
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
        ...settled.state,
        projects: localProject
          ? settled.state.projects.map((project) => (project.id === action.projectId ? serverProject : project))
          : [...settled.state.projects, serverProject],
      };
    }
    case 'retargetProject': {
      const liveProject = state.projects.find((candidate) => candidate.id === action.projectId);
      const retargeted = liveProject
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
      const { project, state: settledState } = settleArriving(state, retargeted, context);
      return {
        ...settledState,
        activeProjectId: state.activeProjectId === action.projectId ? project.id : state.activeProjectId,
        // The same project under its new id: already on screen, not arriving.
        enteredProjectIds: state.enteredProjectIds?.map((id) => (id === action.projectId ? project.id : id)),
        customizing:
          state.customizing?.[action.projectId] === undefined
            ? state.customizing
            : Object.fromEntries(
                Object.entries(state.customizing).map(([id, session]) => [
                  id === action.projectId ? project.id : id,
                  session,
                ])
              ),
        projects: settledState.projects.some((candidate) => candidate.id === action.projectId)
          ? settledState.projects.map((candidate) => (candidate.id === action.projectId ? project : candidate))
          : [...settledState.projects, project],
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
      return {
        ...state,
        autosave: {
          ...(state.autosave.lastSavedAt ? { lastSavedAt: state.autosave.lastSavedAt } : {}),
          error: action.error,
          status: 'pending',
        },
      };
    }
    case 'autosaveNeedsAttention': {
      return { ...state, autosave: { lastSavedAt: action.savedAt, status: 'attention' } };
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
      if (state.notifications.every((notification) => notification.isRead)) {
        return state;
      }

      return {
        ...state,
        notifications: state.notifications.map((notification) =>
          notification.isRead ? notification : { ...notification, isRead: true }
        ),
      };
    }
    case 'markNotificationsRead': {
      const ids = new Set(action.notificationIds);

      if (!state.notifications.some((notification) => !notification.isRead && ids.has(notification.id))) {
        return state;
      }

      return {
        ...state,
        notifications: state.notifications.map((notification) =>
          !notification.isRead && ids.has(notification.id) ? { ...notification, isRead: true } : notification
        ),
      };
    }
    case 'clearNotifications': {
      return state.notifications.length === 0 ? state : { ...state, notifications: [] };
    }
    case 'dismissNotification': {
      const notifications = state.notifications.filter((notification) => notification.id !== action.notificationId);

      return notifications.length === state.notifications.length ? state : { ...state, notifications };
    }
    case 'withdrawNotificationRetry': {
      const isRetry = (candidate: WorkbenchNotificationAction) =>
        candidate.kind === 'retry-queue-item' &&
        candidate.itemIds.length === action.itemIds.length &&
        candidate.itemIds.every((itemId) => action.itemIds.includes(itemId));

      if (!state.notifications.some((notification) => notification.actions?.some(isRetry))) {
        return state;
      }

      return {
        ...state,
        notifications: state.notifications.map((notification) => {
          if (!notification.actions?.some(isRetry)) {
            return notification;
          }

          const { actions, ...rest } = notification;
          const remaining = actions.filter((candidate) => !isRetry(candidate));

          return remaining.length > 0 ? { ...rest, actions: remaining } : rest;
        }),
      };
    }
    case 'restoreNotifications': {
      const currentIds = new Set(state.notifications.map((notification) => notification.id));
      const restored = action.notifications.filter((notification) => !currentIds.has(notification.id));

      return restored.length === 0
        ? state
        : { ...state, notifications: [...state.notifications, ...restored].slice(0, NOTIFICATION_LIMIT) };
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
        createNotification({
          actions: action.actions,
          category: action.category,
          kind: action.kind,
          message: action.message,
          messageKey: action.messageKey,
          messageValues: action.messageValues,
          persistent: action.persistent,
          title: action.title,
          titleKey: action.titleKey,
          titleValues: action.titleValues,
        })
      );
    }
    case 'setActiveProjectSettings': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) => {
        const settings = normalizeProjectSettings({ ...project.settings, ...action.settings });
        // Both sides' keys: an `undefined` in the patch clears a pin, which leaves the key out of `settings`.
        const keys = new Set([...Object.keys(project.settings), ...Object.keys(settings)]) as Set<
          keyof ProjectSettings
        >;

        if ([...keys].every((key) => areProjectSettingValuesEqual(project.settings[key], settings[key]))) {
          return project;
        }

        // Turning Auto-switch back on ends a compare wait, so live frames cannot hide a later comparison.
        const endsCompareWait =
          project.settings.viewerAutoSwitch === 'off' &&
          settings.viewerAutoSwitch !== 'off' &&
          getGalleryCompareState(getWidgetValues(project, 'gallery')) === 'armed';
        const nextProject = { ...project, settings };

        return endsCompareWait
          ? updateProjectWidgetValues(nextProject, 'gallery', (values) => ({ ...values, compareImage: null }))
          : nextProject;
      });
    }
    case 'setScopedDefaults': {
      return Object.is(state.scopedDefaults, action.scopedDefaults)
        ? state
        : { ...state, scopedDefaults: action.scopedDefaults };
    }
  }
};

export type __WorkbenchReducerActionInternal = WorkbenchReducerAction;
