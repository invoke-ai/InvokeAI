import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { QueueCompiledSubmission, QueueHistoryItemStatus } from '@features/queue/contracts';
import type { ProjectGraphState } from '@features/workflow/contracts';
import type {
  CanvasDocumentContractV2,
  CanvasPlacementContract,
  CanvasStateContractV2,
  CanvasStagingCandidateContract,
} from '@workbench/canvas-engine/api';
import type { DeveloperLogNamespace } from '@workbench/diagnostics/contracts';
import type { GraphContract } from '@workbench/graphContracts';
import type { InvocationRoute, InvocationSourceId, ResultDestination } from '@workbench/invocationContracts';
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
  ProjectLayoutState,
  WidgetRegion,
  WidgetRegionState,
} from '@workbench/layoutContracts';
import type {
  GraphHistorySnapshot,
  Project,
  ProjectUndoSnapshot,
  PromptHistoryItem,
  WorkbenchNotification,
  WorkbenchNotificationCategory,
  WorkbenchNotificationKind,
  WorkbenchState,
} from '@workbench/projectContracts';
import type { ProjectRecoveredIdentity } from '@workbench/projects/projectFlush';
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
  getGallerySettings,
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
} from '@features/gallery/contracts';

import type { WorkbenchQueueItem as QueueItem } from './queueHistoryContracts';

import {
  getChangedValueKeys,
  getRouteAfterHighConfidenceEdit,
  isHighConfidenceCanvasEdit,
  isHighConfidenceCanvasEditIntent,
  isHighConfidenceGenerateEdit,
  isHighConfidenceGraphEdit,
  isHighConfidenceUpscaleEdit,
  type CanvasEditIntent,
  type WorkbenchActionOrigin,
} from './autoRoutePolicy';
import { createNewCanvasStateV2, migrateCanvasStateToV2 } from './canvasMigration';
import { applyCanvasProjectMutation, type CanvasProjectMutation } from './canvasProjectMutations';
import { getProjectWidgetValues } from './widgetState';
export { nextLayerName } from './canvasProjectMutations';
import { compileGenerateGraph, resolveGenerateSeed } from '@features/generation/graph';
import {
  addPromptHistoryItem,
  applyProjectPromptDraft,
  cloneGenerateWidgetValues,
  getEffectivePrompts,
  getGenerationModelAvailabilityReasons,
  getPromptDraftFromValues,
  getPromptHistoryItemFromGenerateSettings,
  hasDynamicPromptSyntax,
  migrateProjectPromptDraft,
  normalizeGenerateSettings,
  normalizeGenerateWidgetValues,
  type ProjectPromptDraftPatch,
  removePromptHistoryItem,
  sanitizeBatchCount,
  syncGenerateWidgetValuesWithModels,
} from '@features/generation/settings';
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
import { compileProjectGraph } from '@features/workflow/graph';
import { getInvocationTemplatesSnapshot } from '@features/workflow/react';
import {
  cloneProjectGraph,
  createProjectGraph,
  getProjectGraphUndoLabel,
  normalizeProjectGraph,
  projectGraphReducer,
  type ProjectGraphAction,
} from '@features/workflow/utility';

import {
  getCanvasStagingSlotCount,
  getCanvasStagingSlots,
  getFirstCanvasPlaceholderSlotIndex,
} from './canvasStagingView';
import { cascadeDefaultGeometry, clampSizeToMinimum, nextStackOrder } from './floatingWindows';
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
  cloneFloatingWidgets,
  cloneLayoutPresetWidgetRegions,
  createLayoutPresetSnapshot,
  resolveSavedLayoutPreset,
} from './layoutPresetSnapshots';
import { normalizeWorkbenchQueueHistory } from './queueHistoryNormalization';
import { normalizeProjectSettings } from './settings/store';

type QueueGenerateSnapshot = NonNullable<QueueItem['snapshot']['generate']>;

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
  | { type: 'setRegionWidgetCollapsed'; region: WidgetRegion; isCollapsed: boolean }
  | { type: 'setRegionWidgetSize'; region: WidgetRegion; sizePx: number }
  | { type: 'floatWidget'; instanceId: WidgetInstanceId }
  | { type: 'dockFloatingWidget'; instanceId: WidgetInstanceId }
  | {
      type: 'setFloatingWidgetGeometry';
      instanceId: WidgetInstanceId;
      x: number;
      y: number;
      widthPx: number;
      heightPx: number;
    }
  | { type: 'setFloatingWidgetMode'; instanceId: WidgetInstanceId; mode: FloatingWidgetMode }
  | { type: 'focusFloatingWidget'; instanceId: WidgetInstanceId }
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
  | { type: 'applyProjectGraphAction'; action: ProjectGraphAction }
  | { type: 'replaceProjectGraph'; document: ProjectGraphState; label: string }
  | { type: 'saveProjectGraphSnapshot' }
  | { type: 'restoreProjectGraphSnapshot'; snapshotId: string }
  | { type: 'setProjectGraphLibraryBinding'; libraryWorkflowId: string }
  | { type: 'submitInvocationSnapshot'; backendSupportsCancellation: boolean; models?: readonly ModelConfig[] }
  | {
      type: 'submitResolvedInvocationSnapshot';
      backendSupportsCancellation: boolean;
      /** Expanded positive prompts, resolved by the caller before dispatch. */
      positivePrompts?: string[];
      route: InvocationRoute;
      models?: readonly ModelConfig[];
    }
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
    }
  | { type: 'markQueueItemBackendCancelled'; projectId: string; queueItemId: string; backendItemId: number }
  | { type: 'routeQueueItemResults'; projectId: string; queueItemId: string; images: GeneratedImageContract[] }
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
  | { type: 'setGalleryMultiSelection'; itemKeys: GalleryItemKey[]; primaryItem: GalleryItem; projectId?: string }
  | { type: 'setGalleryCompareImage'; image: GalleryImageItem | null; projectId?: string }
  | { type: 'selectGalleryBoard'; boardId: string; projectId?: string }
  | { type: 'setGalleryView'; galleryView: 'images' | 'assets'; projectId?: string }
  | { type: 'setGallerySearchTerm'; searchTerm: string; projectId?: string }
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
  | {
      type: 'submitCanvasInvocationSnapshot';
      backendSupportsCancellation: boolean;
      canvas: CanvasStateContractV2;
      destination: ResultDestination;
      generate: QueueGenerateSnapshot;
      graph: GraphContract;
      /** Expanded positive prompts, resolved by the caller before dispatch. */
      positivePrompts?: string[];
      projectId: string;
    }
  | { type: 'cancelQueueItem'; queueItemId: string; projectId?: string }
  | { type: 'cancelAllQueueItems'; projectId?: string }
  | { type: 'cancelAllQueueItemsExceptCurrent'; projectId?: string; currentQueueItemId?: string | null }
  | { type: 'clearCompletedQueueItems' }
  | { type: 'undoProjectChange' }
  | { type: 'redoProjectChange' }
  | { type: 'hydrateWorkbench'; state: WorkbenchState }
  | {
      type: 'reconcileProjectConflict';
      projectId: string;
      serverProject: Project;
      recoveredProject: Project;
      recoveredIdentity: ProjectRecoveredIdentity;
    }
  | {
      type: 'reconcileDeletedProject';
      projectId: string;
      recoveredProject: Project;
      recoveredIdentity: ProjectRecoveredIdentity;
    }
  | { type: 'autosaveStarted' }
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
      context?: { error?: string; layerId?: string };
      namespace?: DeveloperLogNamespace;
      projectId?: string;
    }
  | { type: 'setBackendConnectionStatus'; status: WorkbenchState['backendConnection']['status']; error?: string }
  | { type: 'recordNotice'; kind: WorkbenchNotificationKind; title: string; message?: string };

const HISTORY_LIMIT = 40;
export const GRAPH_HISTORY_BYTE_BUDGET = 64 * 1024 * 1024;
const NOTIFICATION_LIMIT = 100;
// Side panels host real widget UIs (gallery grid, generate form); below
// ~350px their toolbars and grids collapse into unusable slivers, so that is
// the floor rather than a merely-rendered 180px. The bottom strip is a
// status row, not a widget host, and keeps its own bounds.
const MIN_PANEL_SIZE_PX = 350;
const MAX_PANEL_SIZE_PX = 520;
const MIN_STATUS_PANEL_SIZE_PX = 96;
const MAX_STATUS_PANEL_SIZE_PX = 420;

/** The resize bounds for a widget region — shared with the resize handles. */
export const getPanelSizeBounds = (region: WidgetRegion): { max: number; min: number } => {
  if (region === 'bottom') {
    return { max: MAX_STATUS_PANEL_SIZE_PX, min: MIN_STATUS_PANEL_SIZE_PX };
  }

  return { max: MAX_PANEL_SIZE_PX, min: MIN_PANEL_SIZE_PX };
};

const now = (): string => new Date().toISOString();

const createId = (prefix: string): string =>
  `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

const createNotification = ({
  category,
  kind,
  message,
  projectId,
  title,
}: {
  category?: WorkbenchNotificationCategory;
  kind: WorkbenchNotificationKind;
  message?: string;
  projectId?: string;
  title: string;
}): WorkbenchNotification => ({
  category,
  createdAt: now(),
  id: createId('notification'),
  isRead: false,
  kind,
  message,
  projectId,
  title,
});

const addNotification = (state: WorkbenchState, notification: WorkbenchNotification): WorkbenchState => {
  const [newest, ...rest] = state.notifications;

  // Coalesce an exact repeat of the newest ERROR notification into an
  // occurrence bump on the SAME id, instead of stacking a new one — the
  // toaster dedupes toasts by id, so a repeat then stops re-toasting for free
  // (e.g. an ambient retry failing the same way every cycle). Restricted to
  // errors: non-error kinds (e.g. "Invocation queued") are routine, repeat
  // actions that must each surface their own toast.
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

/** Adds the "Invocation queued" notice iff the reduction actually grew that project's queue. */
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

  return addNotification(
    nextState,
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
        })),
        nodes: Object.fromEntries(Object.entries(graph.backendGraph.nodes).map(([id, node]) => [id, { ...node }])),
      }
    : undefined,
  edges: graph.edges.map((edge) => ({ ...edge })),
  nodes: graph.nodes.map((node) => ({ ...node, inputs: { ...node.inputs } })),
});

const cloneQueueGenerateSnapshot = (generate: QueueGenerateSnapshot): QueueGenerateSnapshot => ({
  negativePromptNodeId: generate.negativePromptNodeId,
  positivePromptNodeId: generate.positivePromptNodeId,
  seedNodeId: generate.seedNodeId,
  values: cloneGenerateWidgetValues(generate.values),
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
  document: Pick<CanvasDocumentContractV2, 'height' | 'width'>
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
  document: Pick<CanvasDocumentContractV2, 'height' | 'width'>,
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
  canvas: CanvasStateContractV2,
  pendingImages: CanvasStagingCandidateContract[]
): CanvasStateContractV2 => ({
  ...canvas,
  stagingArea: {
    ...canvas.stagingArea,
    pendingImages,
  },
});

const getCanvasStagingCandidateSlotIndex = (
  project: Project,
  pendingImages: CanvasStagingCandidateContract[],
  target: CanvasStagingCandidateContract | undefined
): number => {
  if (!target) {
    return -1;
  }

  return getCanvasStagingSlots(
    getCanvasWithPendingImages(project.canvas, pendingImages),
    project.queue.items
  ).findIndex(
    (slot) =>
      slot.kind === 'candidate' &&
      slot.candidate.sourceQueueItemId === target.sourceQueueItemId &&
      slot.candidate.imageName === target.imageName
  );
};

const resolveStagingSelectionIndexForSlots = ({
  incomingImages,
  pendingImages,
  project,
  slotCount,
}: {
  incomingImages: CanvasStagingCandidateContract[];
  pendingImages: CanvasStagingCandidateContract[];
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

  if (incomingImages.length === 0) {
    return clampStagedImageIndex(project.canvas.stagingArea.selectedImageIndex, slotCount);
  }

  const selectedImage =
    project.canvas.stagingArea.autoSwitchMode === 'latest'
      ? (pendingImages[pendingImages.length - 1] ?? incomingImages[incomingImages.length - 1])
      : incomingImages[0];
  const selectedSlotIndex = getCanvasStagingCandidateSlotIndex(project, pendingImages, selectedImage);

  return selectedSlotIndex === -1
    ? clampStagedImageIndex(project.canvas.stagingArea.selectedImageIndex, slotCount)
    : selectedSlotIndex;
};

const stageCanvasResultImages = (
  project: Project,
  queueItemId: string,
  images: GeneratedImageContract[],
  sourceBackendItemIds?: readonly (number | undefined)[]
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
  const candidateKey = `${appendedCandidate.sourceQueueItemId}:${appendedCandidate.imageName}`;
  const slots = getCanvasStagingSlots(getCanvasWithPendingImages(project.canvas, pendingImages), project.queue.items);
  const selectedImageIndex = slots
    .map((slot) => (slot.kind === 'candidate' ? `${slot.candidate.sourceQueueItemId}:${slot.candidate.imageName}` : ''))
    .lastIndexOf(candidateKey);

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

const clampCanvasStagingSelection = (project: Project): Project => {
  const slotCount = getCanvasStagingSlotCount(project.canvas, project.queue.items);
  const placeholderIndex =
    project.canvas.stagingArea.autoSwitchMode === 'progress'
      ? getFirstCanvasPlaceholderSlotIndex(project.canvas, project.queue.items)
      : -1;
  const selectedImageIndex =
    placeholderIndex === -1
      ? clampStagedImageIndex(project.canvas.stagingArea.selectedImageIndex, slotCount)
      : placeholderIndex;
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
 * `migrateCanvasStateToV2` first (see `normalizeWorkbenchProject`).
 */
const cloneCanvas = (canvas: CanvasStateContractV2): CanvasStateContractV2 => {
  const document = structuredClone(canvas.document);

  return {
    version: 2,
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

const cloneQueueWidgetState = (widgetState: WidgetStateContract, typeId: WidgetTypeId): WidgetStateContract => {
  const state = cloneWidgetState(widgetState);

  if (typeId === 'gallery') {
    delete state.values.recentImages;
  }

  return state;
};

const cloneWidgetInstances = (
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>
): Record<WidgetInstanceId, WidgetInstanceContract> =>
  Object.fromEntries(
    Object.entries({ ...createWidgetInstances(), ...widgetInstances }).map(([instanceId, widgetInstance]) => [
      instanceId,
      cloneWidgetInstance(widgetInstance),
    ])
  );

const cloneQueueWidgetInstances = (
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>
): Record<WidgetInstanceId, WidgetInstanceContract> =>
  Object.fromEntries(
    Object.entries({ ...createWidgetInstances(), ...widgetInstances }).map(([instanceId, widgetInstance]) => [
      instanceId,
      {
        ...widgetInstance,
        state: cloneQueueWidgetState(widgetInstance.state, widgetInstance.typeId),
      },
    ])
  );

const getWidgetStatesSnapshot = (widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>): WidgetStateMap => {
  const widgetStates: WidgetStateMap = {};

  for (const widgetInstance of Object.values(widgetInstances)) {
    widgetStates[widgetInstance.typeId] ??= cloneQueueWidgetState(widgetInstance.state, widgetInstance.typeId);
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

const cloneWidgetRegions = (
  widgetRegions: Record<WidgetRegion, WidgetRegionState>
): Record<WidgetRegion, WidgetRegionState> => ({
  center: {
    ...widgetRegions.center,
    instanceIds: [...widgetRegions.center.instanceIds],
  },
  left: {
    ...widgetRegions.left,
    instanceIds: [...widgetRegions.left.instanceIds],
  },
  right: {
    ...widgetRegions.right,
    instanceIds: [...widgetRegions.right.instanceIds],
  },
  bottom: {
    ...widgetRegions.bottom,
    instanceIds: [...widgetRegions.bottom.instanceIds],
  },
});

const cloneWidgetGraphs = (widgetGraphs: Project['widgetGraphs']): Project['widgetGraphs'] =>
  Object.fromEntries(Object.entries(widgetGraphs).map(([key, graph]) => [key, graph ? cloneGraph(graph) : graph]));

// Canvas is intentionally absent from undo snapshots: the canvas rendering
// engine owns its own pixel-patch history, so project-level undo/redo neither
// snapshots nor restores canvas — `restoreUndoSnapshot` passes the live
// `project.canvas` straight through via the `...project` spread.
const createUndoSnapshot = (
  project: Project,
  projectGraph = cloneProjectGraph(project.projectGraph)
): ProjectUndoSnapshot => ({
  floatingWidgets: project.floatingWidgets ? { ...project.floatingWidgets } : undefined,
  invocation: { ...project.invocation },
  layout: { ...project.layout, panels: { ...project.layout.panels } },
  projectGraph,
  widgetGraphs: cloneWidgetGraphs(project.widgetGraphs),
  widgetInstances: cloneWidgetInstances(project.widgetInstances),
  widgetRegions: cloneWidgetRegions(project.widgetRegions),
});

const restoreUndoSnapshot = (project: Project, snapshot: ProjectUndoSnapshot): Project => ({
  ...project,
  // Restored WITH widgetRegions — they are one placement fact, and restoring
  // one without the other can double-render or orphan a floated instance.
  floatingWidgets: snapshot.floatingWidgets ? { ...snapshot.floatingWidgets } : undefined,
  invocation: { ...snapshot.invocation },
  layout: { ...snapshot.layout, panels: { ...snapshot.layout.panels } },
  projectGraph: cloneProjectGraph(normalizeProjectGraph(snapshot.projectGraph)),
  widgetGraphs: cloneWidgetGraphs(snapshot.widgetGraphs),
  widgetInstances: cloneWidgetInstances(snapshot.widgetInstances),
  widgetRegions: cloneWidgetRegions(snapshot.widgetRegions),
});

const UTF8_ENCODER = new TextEncoder();

const getGraphHistorySnapshotBytes = (snapshot: GraphHistorySnapshot): number => {
  const { retainedBytes: _retainedBytes, ...serialized } = snapshot;
  let retainedBytes = 0;

  // Include the metadata field itself in the serialized-size budget. Its digit
  // count can change the answer, so converge on the stable JSON byte length.
  for (;;) {
    const nextRetainedBytes = UTF8_ENCODER.encode(JSON.stringify({ ...serialized, retainedBytes })).byteLength;
    if (nextRetainedBytes === retainedBytes) {
      return retainedBytes;
    }
    retainedBytes = nextRetainedBytes;
  }
};

type MeasuredGraphHistorySnapshot = GraphHistorySnapshot & { retainedBytes: number };

const withRetainedBytes = (snapshot: GraphHistorySnapshot): MeasuredGraphHistorySnapshot => ({
  ...snapshot,
  retainedBytes: getGraphHistorySnapshotBytes(snapshot),
});

/** Trims state-owned snapshots without reserializing retained history. */
const trimMeasuredGraphHistory = (snapshots: readonly GraphHistorySnapshot[]): GraphHistorySnapshot[] => {
  const history: GraphHistorySnapshot[] = [];
  let retainedBytes = 0;

  for (const snapshot of snapshots) {
    if (history.length >= HISTORY_LIMIT) {
      break;
    }

    const snapshotBytes = snapshot.retainedBytes;
    if (typeof snapshotBytes !== 'number' || !Number.isFinite(snapshotBytes) || snapshotBytes < 0) {
      continue;
    }

    if (snapshotBytes > GRAPH_HISTORY_BYTE_BUDGET || retainedBytes + snapshotBytes > GRAPH_HISTORY_BYTE_BUDGET) {
      continue;
    }

    retainedBytes += snapshotBytes;
    history.push(snapshot);
  }

  return history;
};

export const normalizeGraphHistory = (value: unknown): GraphHistorySnapshot[] => {
  if (!Array.isArray(value)) {
    return [];
  }

  const measuredHistory: MeasuredGraphHistorySnapshot[] = [];

  for (const item of value) {
    if (!item || typeof item !== 'object') {
      continue;
    }

    const snapshot = item as GraphHistorySnapshot;
    if (
      typeof snapshot.id !== 'string' ||
      typeof snapshot.createdAt !== 'string' ||
      typeof snapshot.label !== 'string'
    ) {
      continue;
    }

    // Persisted metadata is untrusted. Always derive the retained size from the
    // actual snapshot so a forged low count cannot bypass the load-time budget
    // and a stale high count cannot discard valid history.
    measuredHistory.push(withRetainedBytes(snapshot));
  }

  return trimMeasuredGraphHistory(measuredHistory);
};

const prependGraphHistory = (
  history: readonly GraphHistorySnapshot[],
  snapshot: MeasuredGraphHistorySnapshot
): GraphHistorySnapshot[] => trimMeasuredGraphHistory([snapshot, ...history]);

const createGraphHistorySnapshot = (label: string, graph: GraphContract): MeasuredGraphHistorySnapshot =>
  withRetainedBytes({
    createdAt: now(),
    graph: cloneGraph(graph),
    id: createId('graph-history'),
    label,
  });

/** A restorable history entry carrying the editable workflow document. */
const createDocumentHistorySnapshot = (
  label: string,
  document: ProjectGraphState,
  cloneDocument = true
): MeasuredGraphHistorySnapshot =>
  withRetainedBytes({
    createdAt: now(),
    document: cloneDocument ? cloneProjectGraph(document) : document,
    id: createId('graph-history'),
    label,
  });

const pushUndo = (project: Project, label: string, projectGraph?: ProjectGraphState): Project => ({
  ...project,
  undoRedo: {
    future: [],
    past: [
      ...project.undoRedo.past,
      {
        createdAt: now(),
        id: createId('undo'),
        label,
        project: createUndoSnapshot(project, projectGraph),
      },
    ].slice(-HISTORY_LIMIT),
  },
});

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
  'version-status': { id: 'version-status', label: 'Version', values: {}, version: 1 },
  workflow: { graphId: 'workflow-graph', id: 'workflow', label: 'Workflow', values: {}, version: 1 },
  upscale: { graphId: 'upscale-graph', id: 'upscale', label: 'Upscale', values: {}, version: 1 },
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
  layers: 'layers',
  notifications: 'notifications',
  preview: 'preview',
  project: 'project',
  queue: 'queue',
  'server-status': 'server-status',
  'version-status': 'version-status',
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
  if (leftRegion.instanceIds.includes('upscale')) {
    return leftRegion;
  }

  const legacyMatch = LEGACY_DEFAULT_LEFT_REGION_WIDGET_IDS.some(
    (ids) =>
      ids.length === leftRegion.instanceIds.length && ids.every((id, index) => leftRegion.instanceIds[index] === id)
  );

  if (!legacyMatch) {
    return leftRegion;
  }

  const galleryIndex = leftRegion.instanceIds.indexOf('gallery');
  const instanceIds = [...leftRegion.instanceIds];

  instanceIds.splice(galleryIndex === -1 ? instanceIds.length : galleryIndex, 0, 'upscale');

  return { ...leftRegion, instanceIds };
};

// Every right rail this app has shipped as a default. A project persisted with
// one of these exactly is an untouched default rather than a customization, so
// it adopts the current curated rail wholesale.
//
// Adopting beats splicing the new widget in: the curated presets are the only
// arrangements a rail can hold without reading as drifted, and a spliced rail
// is by construction not one of them — it would show the unsaved-changes dot
// and offer to revert a layout nobody edited.
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

// The shipped bottom-region default before 'queue-status' was added — a
// persisted project whose bottom rail matches this exactly is still running
// the pre-branch defaults, so it should pick up the new widget the same way
// a fresh project would.
//
// A rail whose 'queue-status' was floated back out matches this shape too, so
// the migration re-docks it on every load. That is left to
// `reconcileFloatingWidgets`, which runs on this region's output and drops any
// instance holding a floating window — the same contract the other rail
// migrations here rely on.
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
  const activeInstanceId = centerRegion?.activeInstanceId ?? getCenterWidgetIdFromViewId(fallbackCenterViewId);
  const instanceIds = centerRegion?.instanceIds.length ? centerRegion.instanceIds : defaultCenterRegion.instanceIds;
  const normalizedActiveInstanceId = instanceIds.includes(activeInstanceId) ? activeInstanceId : instanceIds[0];

  return {
    ...defaultCenterRegion,
    ...centerRegion,
    activeInstanceId: normalizedActiveInstanceId,
    instanceIds,
    isCollapsed: false,
  };
};

const WIDGET_REGION_IDS: WidgetRegion[] = ['left', 'right', 'bottom', 'center'];
const FLOATING_WIDGET_MODES: FloatingWidgetMode[] = ['windowed', 'maximized', 'shaded'];

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

const isFloatingWidgetMode = (value: unknown): value is FloatingWidgetMode =>
  FLOATING_WIDGET_MODES.includes(value as FloatingWidgetMode);

const isWidgetRegionId = (value: unknown): value is WidgetRegion => WIDGET_REGION_IDS.includes(value as WidgetRegion);

/**
 * Put a docking widget back where it was, not on the end.
 *
 * The rail is an ordered tab strip, so appending turned float-then-dock — a
 * gesture that reads as undoing the float — into a permanent reordering, which
 * then registered as drift from the preset. The rail may have changed while the
 * window was open, so the remembered index is clamped rather than trusted.
 */
const insertAtReturnIndex = (
  instanceIds: WidgetInstanceId[],
  instanceId: WidgetInstanceId,
  returnIndex: number | undefined
): WidgetInstanceId[] => {
  const next = [...instanceIds];

  next.splice(
    isFiniteNumber(returnIndex) && returnIndex >= 0 ? Math.min(Math.floor(returnIndex), next.length) : next.length,
    0,
    instanceId
  );

  return next;
};

/**
 * Persisted floating windows are an unsafe-cast boundary like every other
 * sub-shape here: an entry naming a region that does not exist crashes the
 * reducer the moment it is docked, and non-numeric geometry reaches the
 * window's fixed-position CSS. Anything malformed is dropped, so the widget
 * reappears docked rather than not at all.
 */
const normalizeFloatingWidgets = (
  value: unknown,
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>
): Record<WidgetInstanceId, FloatingWidgetState> | undefined => {
  if (!value || typeof value !== 'object') {
    return undefined;
  }

  const floatingWidgets: Record<WidgetInstanceId, FloatingWidgetState> = {};

  for (const [instanceId, entry] of Object.entries(value as Record<string, unknown>)) {
    if (!entry || typeof entry !== 'object' || !widgetInstances[instanceId]) {
      continue;
    }

    const state = entry as Partial<FloatingWidgetState>;

    if (
      !isFiniteNumber(state.x) ||
      !isFiniteNumber(state.y) ||
      !isFiniteNumber(state.widthPx) ||
      !isFiniteNumber(state.heightPx) ||
      !isFiniteNumber(state.stackOrder) ||
      !isFloatingWidgetMode(state.mode) ||
      !isWidgetRegionId(state.returnRegion)
    ) {
      continue;
    }

    floatingWidgets[instanceId] = {
      ...clampSizeToMinimum({ heightPx: state.heightPx, widthPx: state.widthPx, x: state.x, y: state.y }),
      mode: state.mode,
      // Carried explicitly, like every other field: this rebuilds the entry
      // rather than spreading it, so anything not named here is dropped. A
      // nonsensical index is simply omitted — docking falls back to appending.
      ...(isFiniteNumber(state.returnIndex) && state.returnIndex >= 0
        ? { returnIndex: Math.floor(state.returnIndex) }
        : {}),
      returnRegion: state.returnRegion,
      stackOrder: state.stackOrder,
    };
  }

  return Object.keys(floatingWidgets).length > 0 ? floatingWidgets : undefined;
};

/**
 * An instance renders either in a region or in a floating window, never both.
 *
 * The region migrations above rebuild a rail that reads as an untouched default
 * — and a rail missing a floated widget is exactly that shape — so on every
 * reload they hand back a widget the person had floated. Floating wins: it is
 * the deliberate act, while the region entry is the migration's guess.
 *
 * The center region is the exception, because it must always hold a view. If
 * honouring the floating entries would empty it, they lose and the widget
 * stays docked.
 */
const reconcileFloatingWidgets = (
  widgetRegions: Record<WidgetRegion, WidgetRegionState>,
  floatingWidgets: Record<WidgetInstanceId, FloatingWidgetState> | undefined
): {
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
  floatingWidgets: Record<WidgetInstanceId, FloatingWidgetState> | undefined;
} => {
  if (!floatingWidgets) {
    return { floatingWidgets, widgetRegions };
  }

  let remainingFloating = floatingWidgets;
  const reconciledRegions = { ...widgetRegions };

  for (const regionId of WIDGET_REGION_IDS) {
    const region = reconciledRegions[regionId];
    const instanceIds = region.instanceIds.filter((instanceId) => !remainingFloating[instanceId]);

    if (instanceIds.length === region.instanceIds.length) {
      continue;
    }

    if (regionId === 'center' && instanceIds.length === 0) {
      remainingFloating = Object.fromEntries(
        Object.entries(remainingFloating).filter(([instanceId]) => !region.instanceIds.includes(instanceId))
      );
      continue;
    }

    reconciledRegions[regionId] = {
      ...region,
      activeInstanceId: instanceIds.includes(region.activeInstanceId)
        ? region.activeInstanceId
        : (instanceIds[0] ?? region.activeInstanceId),
      instanceIds,
      isCollapsed: instanceIds.length === 0 ? regionId !== 'center' : region.isCollapsed,
    };
  }

  return {
    floatingWidgets: Object.keys(remainingFloating).length > 0 ? remainingFloating : undefined,
    widgetRegions: reconciledRegions,
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

export const normalizeWorkbenchProject = (project: Project): Project => {
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

  if (leftRegion.instanceIds.includes('upscale') && !widgetInstances.upscale) {
    widgetInstances.upscale = createWidgetInstance('upscale');
  }

  if (bottomRegion.instanceIds.includes('queue-status') && !widgetInstances['queue-status']) {
    widgetInstances['queue-status'] = createWidgetInstance('queue-status');
  }

  for (const [instanceId, instance] of Object.entries(widgetInstances)) {
    if (instance.typeId !== 'gallery' || !('recentImages' in instance.state.values)) {
      continue;
    }

    widgetInstances[instanceId] = {
      ...instance,
      state: {
        ...instance.state,
        values: {
          ...instance.state.values,
          recentImages: getBoundedRecentImages(instance.state.values.recentImages),
        },
      },
    };
  }

  const canvas = cloneCanvas(migrateCanvasStateToV2(project.canvas));
  const placement = reconcileFloatingWidgets(
    {
      left: leftRegion,
      right: ensureRightRegion(legacyWidgetRegions?.right ?? legacyWidgetRegions?.['right-panel']),
      bottom: bottomRegion,
      center: ensureCenterRegion(legacyWidgetRegions?.center, project.layout.centerViewId),
    },
    normalizeFloatingWidgets((project as Partial<Project>).floatingWidgets, widgetInstances)
  );

  return {
    ...project,
    // `project` may come straight from persisted storage (an unsafe cast boundary), so its
    // canvas can still be v1-shaped, malformed, or missing — migrate before cloning.
    canvas,
    floatingWidgets: placement.floatingWidgets,
    graphHistory: normalizeGraphHistory((project as Partial<Project>).graphHistory),
    // Built-in preset ids were renamed for the three-preset model; a project
    // saved under an old id must still resolve to the arrangement it names,
    // otherwise every restored project reads as drifted from Compose.
    layout: { ...project.layout, presetId: resolveLayoutPresetId(project.layout.presetId) },
    projectGraph: normalizeProjectGraph(project.projectGraph),
    promptHistory: normalizePromptHistory((project as Partial<Project>).promptHistory),
    queue: normalizeWorkbenchQueueHistory(project.queue, { canvas, widgetInstances }),
    settings: normalizeProjectSettings(project.settings),
    widgetRegions: placement.widgetRegions,
    widgetInstances,
  };
};

/**
 * Write the server's board id into a *hydrated* project's gallery state.
 *
 * Patching the document before rehydration is not enough on its own. A project saved by a build
 * that never opened its Gallery widget — or one whose document predates widget instances entirely —
 * has no gallery values for the patch to land in, and the instance the reducer creates during
 * normalization arrives afterwards, empty. Such a project would then show the placeholder board row
 * forever and route nothing at its own board, which is the one thing the server is authoritative
 * about.
 *
 * Applied after normalization, so the instance exists. A project whose layout has no gallery widget
 * at all is returned untouched: there is nothing to tell.
 */
export const withAuthoritativeProjectBoard = (project: Project, boardId: string): Project =>
  updateProjectWidgetValues(project, 'gallery', (values) =>
    values.projectBoardId === boardId ? values : { ...values, projectBoardId: boardId }
  );

/**
 * The project a recovery fork should become, preferring live content over the snapshot.
 *
 * The fork is serialized when the save begins, so anything typed since is newer than it. Adopting
 * the snapshot would delete precisely the edits the fork exists to rescue, in the case the
 * mechanism most often fires: a save is stale exactly when a keystroke landed mid-flight.
 *
 * So the live project is re-labelled instead. The server-side fork already holds the older document
 * under this identity, so the next push sees a difference and sends the current content up its
 * revision chain — nothing lost, nothing to merge.
 *
 * The snapshot still wins when there is no live project (a tab closed mid-save), because then it is
 * the only local copy of that work.
 */
const recoverProjectUnderNewIdentity = (
  localProject: Project | undefined,
  snapshotProject: Project,
  identity: ProjectRecoveredIdentity
): Project =>
  normalizeWorkbenchProject(
    localProject
      ? {
          ...localProject,
          id: identity.id,
          name: identity.name,
          recoveredAt: identity.recoveredAt,
          recoveryOf: identity.recoveryOf,
        }
      : snapshotProject
  );

export const clampPanelSize = (region: WidgetRegion, sizePx: number): number => {
  const { max, min } = getPanelSizeBounds(region);

  return Math.min(max, Math.max(min, sizePx));
};

const createCanvasState = (): CanvasStateContractV2 => createNewCanvasStateV2();

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
      graphHistory: [],
      id,
      invocation: getInvocationAfterLayoutPreset(defaultInvocationRoute, preset),
      layout: { ...defaultLayoutPreset.snapshot.layout, panels: { ...defaultLayoutPreset.snapshot.layout.panels } },
      name: `Project Name #${index}`,
      promptHistory: [],
      projectGraph: createProjectGraph(`${id}-graph`),
      queue: { items: [] },
      settings: normalizeProjectSettings(),
      undoRedo: { future: [], past: [] },
      widgetGraphs: {},
      widgetInstances: createWidgetInstances(),
      widgetRegions: createWidgetRegions(),
    },
    preset
  );

const getNextProjectIndex = (projects: Project[]): number => {
  const usedIndices = projects.map((project) => Number(project.name.match(/#(\d+)$/)?.[1] ?? 0));

  return Math.max(0, ...usedIndices) + 1;
};

/**
 * A fresh, never-saved project. Ids carry entropy rather than an index so a
 * draft can never collide with a project that already exists on the server
 * (which an autosave would then silently overwrite).
 */
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

const getNextInstanceId = (region: WidgetRegionState, instanceId: WidgetInstanceId): WidgetInstanceId | null => {
  if (region.activeInstanceId !== instanceId) {
    return region.activeInstanceId;
  }

  return region.instanceIds.find((enabledInstanceId) => enabledInstanceId !== instanceId) ?? null;
};

const insertAt = <Value>(values: Value[], value: Value, index: number): Value[] => {
  const nextValues = values.filter((candidate) => candidate !== value);
  const nextIndex = Math.min(nextValues.length, Math.max(0, index));

  nextValues.splice(nextIndex, 0, value);

  return nextValues;
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

const cloneLayoutPresetSnapshot = (snapshot: LayoutPresetSnapshot): LayoutPresetSnapshot => ({
  // Every account preset is rebuilt through here on load, so a field missing
  // from this clone is a field the preset silently loses on the next reload.
  ...(snapshot.floatingWidgets ? { floatingWidgets: cloneFloatingWidgets(snapshot.floatingWidgets) } : {}),
  layout: { ...snapshot.layout, panels: { ...snapshot.layout.panels } },
  widgetInstances: Object.fromEntries(
    Object.entries(snapshot.widgetInstances).map(([instanceId, instance]) => [instanceId, { ...instance }])
  ),
  widgetRegions: cloneLayoutPresetWidgetRegions(snapshot.widgetRegions),
});

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

  return (
    typeof record.activeInstanceId === 'string' &&
    record.activeInstanceId.length > 0 &&
    Array.isArray(instanceIds) &&
    instanceIds.every((instanceId) => typeof instanceId === 'string' && instanceId in widgetInstances) &&
    new Set(instanceIds).size === instanceIds.length &&
    record.activeInstanceId in widgetInstances &&
    (instanceIds.length === 0 || instanceIds.includes(record.activeInstanceId)) &&
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

const normalizeWorkbenchState = (state: WorkbenchState): WorkbenchState => ({
  ...state,
  backendConnection: { status: 'connecting' },
  // Built explicitly: legacy snapshots carried preferences inside the account
  // (they live in the settings store now) and must not resurface here.
  account: normalizeWorkbenchAccount(state.account),
  notifications: [],
  projects: state.projects.map(normalizeWorkbenchProject),
});

const updateActiveLayout = (
  state: WorkbenchState,
  getLayout: (layout: ProjectLayoutState) => ProjectLayoutState
): WorkbenchState =>
  updateActiveProject(state, (project) => {
    const nextProject = pushUndo(project, 'Update layout');

    return {
      ...nextProject,
      events: [
        {
          createdAt: now(),
          id: createId('event'),
          summary: 'Updated active layout',
          type: 'layout-updated',
        },
        ...nextProject.events,
      ],
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
  const snapshot = preset.snapshot;
  const widgetInstances = { ...project.widgetInstances };

  for (const instance of Object.values(snapshot.widgetInstances)) {
    widgetInstances[instance.id] = widgetInstances[instance.id]
      ? { ...widgetInstances[instance.id], title: instance.title }
      : createWidgetInstance(instance.typeId, instance.id);
  }

  // A preset is a full placement reset, so the project's own floating windows
  // go: keeping one would double-render whatever the preset docks. The
  // preset's are restored in their place — a preset saved while a widget
  // floated has it in no region, and dropping them both would leave the
  // instance nowhere at all. Preset bodies reach us from account storage
  // without passing through `normalizeWorkbenchProject`, so they are validated
  // and reconciled here on the same terms as a persisted project.
  const placement = reconcileFloatingWidgets(
    cloneLayoutPresetWidgetRegions(snapshot.widgetRegions),
    normalizeFloatingWidgets(snapshot.floatingWidgets, widgetInstances)
  );

  return {
    ...project,
    floatingWidgets: placement.floatingWidgets,
    layout: {
      ...snapshot.layout,
      panels: { ...snapshot.layout.panels },
      presetId: preset.id,
    },
    widgetInstances,
    widgetRegions: placement.widgetRegions,
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
      events: [
        {
          createdAt: now(),
          id: createId('event'),
          summary: 'Updated active layout',
          type: 'layout-updated',
        },
        ...nextProject.events,
      ],
      invocation: applyDefaultRoute
        ? getInvocationAfterLayoutPreset(nextProject.invocation, preset)
        : nextLayoutProject.invocation,
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
      events: [
        {
          createdAt: now(),
          id: createId('event'),
          summary: 'Updated invocation source or destination',
          type: 'invocation-updated',
        },
        ...nextProject.events,
      ],
      invocation: getInvocation(project.invocation),
    };
  });

/**
 * Applies the auto-switch route rule to a project after a high-confidence
 * edit. Deliberately no undo entry or event: the route change rides the
 * edit's own project update, matching the workflow auto-source precedent.
 */
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

/**
 * The generate widget doubles as Canvas's parameter panel — canvas invocations
 * compile from generate values (prepareCanvasInvocation) and canvasDimsSync
 * mirrors generate dims onto the bbox — so generate edits never steal the
 * route from an active canvas source.
 */
const applyAutoRouteForGenerateEdit = (project: Project, context: WorkbenchReducerContext): Project =>
  project.invocation.sourceId === 'canvas' ? project : applyAutoRouteForEdit(project, 'generate', context);

const compileInvocationSnapshot = (
  project: Project,
  route: InvocationRoute,
  models?: readonly ModelConfig[]
): { graph: GraphContract; widgetStates: WidgetStateMap } | null => {
  const widgetStates = getWidgetStatesSnapshot(project.widgetInstances);

  if (route.sourceId === 'workflow') {
    // Compiles the workflow document into an immutable snapshot. Templates are
    // read imperatively; route validation already guaranteed they are loaded.
    const templatesSnapshot = getInvocationTemplatesSnapshot();

    if (templatesSnapshot.status !== 'loaded') {
      return null;
    }

    return { graph: compileProjectGraph(project.projectGraph, templatesSnapshot.templates), widgetStates };
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
    const compiledGraph = compileUpscaleGraph(resolvedValues, route.destination, project.settings).graph;

    widgetStates.upscale = {
      ...widgetStates.upscale,
      graphId: compiledGraph.id,
      values: { ...cloneUpscaleWidgetValues(resolvedValues) },
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
    project.settings
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
            paginationMode: 'infinite',
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

    didChange ||= withoutUpscaleInput !== project;
    return withoutUpscaleInput;
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

    if (!selectedBoardWasDeleted && !projectBoardWasDeleted) {
      return values;
    }

    return {
      ...values,
      ...(selectedBoardWasDeleted ? { galleryPage: 0, selectedBoardId: 'none' } : {}),
      ...(projectBoardWasDeleted ? { projectBoardId: null } : {}),
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
      getValues
    )
  );

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

const isCancellableQueueItem = (item: QueueItem): boolean =>
  item.cancellable && (item.status === 'pending' || item.status === 'running');

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
  const shouldSelectIncomingImage =
    project.settings.showProgressImagesInViewer || typeof galleryValues.selectedImageName !== 'string';
  const nextSelectedImage = shouldSelectIncomingImage ? newImages[0] : undefined;
  const nextSelectedItem = nextSelectedImage ? legacyGeneratedImageToGalleryItem(nextSelectedImage) : undefined;
  const nextSelectedItemKey = nextSelectedItem ? toGalleryItemKey(nextSelectedItem) : undefined;
  const gallerySettings = getGallerySettings(galleryValues);

  return updateProjectWidgetValues(project, 'gallery', () => ({
    ...galleryValues,
    recentImages: getBoundedRecentImages([...newImages, ...previousImages]),
    selectedImage: nextSelectedItem ?? galleryValues.selectedImage,
    selectedImageName: nextSelectedItemKey ?? galleryValues.selectedImageName,
    selectedImageNames: nextSelectedItemKey
      ? [nextSelectedItemKey]
      : getPersistedSelectedGalleryItemKeys(galleryValues),
    ...(nextSelectedImage
      ? {
          selectedImagePage: 0,
          selectedImageQuery: {
            boardId: nextSelectedImage.boardId,
            galleryView: nextSelectedImage.imageCategory === 'general' ? 'images' : 'assets',
            imageOrderDir: gallerySettings.imageOrderDir,
            page: 0,
            paginationMode: 'infinite',
            searchTerm: '',
          },
        }
      : {}),
  }));
};

const routeQueueItemPartialResults = (
  project: Project,
  queueItemId: string,
  backendItemId: number,
  images: GeneratedImageContract[]
): Project => {
  const queueItem = project.queue.items.find((item) => item.id === queueItemId);
  const destination = queueItem?.snapshot.destination ?? project.invocation.destination;
  const nextProject = updateQueueItem(project, queueItemId, (item) => ({
    ...item,
    completedBackendItemIds: item.completedBackendItemIds?.includes(backendItemId)
      ? item.completedBackendItemIds
      : [...(item.completedBackendItemIds ?? []), backendItemId],
    resultImages: mergeImageResults(item.resultImages, images),
  }));

  if (destination === 'gallery') {
    return updateGalleryWithResultImages(nextProject, images);
  }

  return clampCanvasStagingSelection(
    stageCanvasResultImages(
      nextProject,
      queueItemId,
      images,
      images.map(() => backendItemId)
    )
  );
};

const routeQueueItemResults = (project: Project, queueItemId: string, images: GeneratedImageContract[]): Project => {
  const queueItem = project.queue.items.find((item) => item.id === queueItemId);
  const destination = queueItem?.snapshot.destination ?? project.invocation.destination;
  const nextProject = updateQueueItem(project, queueItemId, (item) => ({
    ...item,
    completedBackendItemIds: item.backendItemIds
      ? item.backendItemIds.filter((backendItemId) => !item.cancelledBackendItemIds?.includes(backendItemId))
      : item.completedBackendItemIds,
    resultImages: images,
    status: 'completed',
  }));

  if (destination === 'gallery') {
    return updateGalleryWithResultImages(nextProject, images);
  }

  // A canvas generation belongs to the canvas SESSION it was submitted against,
  // identified by `documentRevision` (bumped only on wholesale swaps — new canvas,
  // snapshot restore, project sync — never on ordinary edits, and captured in the
  // queue item's canvas snapshot at submit time). If a fresh session started while
  // this generation was in flight, its results belong to a document that no longer
  // exists; routing them would resurrect staging on the brand-new canvas (F2). Keep
  // the completed status, but drop the staged candidates.
  if (queueItem && queueItem.snapshot.canvas.documentRevision !== nextProject.canvas.documentRevision) {
    return nextProject;
  }

  const sourceBackendItemIds = queueItem?.backendItemIds?.filter(
    (backendItemId) => !queueItem.cancelledBackendItemIds?.includes(backendItemId)
  );

  return stageCanvasResultImages(nextProject, queueItemId, images, sourceBackendItemIds);
};

/**
 * Enqueues an already-compiled graph snapshot. Shared by the route-validated
 * `submitInvocationSnapshot` and the canvas engine's `submitCanvasInvocationSnapshot`,
 * whose graph is compiled asynchronously outside the reducer.
 */
const enqueueCompiledSnapshot = (
  project: Project,
  route: InvocationRoute,
  compiled: {
    generate?: QueueGenerateSnapshot;
    graph: GraphContract;
    positivePrompts?: string[];
    widgetStates: WidgetStateMap;
  },
  backendSupportsCancellation: boolean,
  canvasSnapshot?: CanvasStateContractV2
): Project => {
  const submittedAt = now();
  const queueItemId = createId('queue-item');
  const { generate, graph } = compiled;
  const widgetStates = Object.fromEntries(
    Object.entries(applyQueueGenerateSnapshotToWidgetStates(compiled.widgetStates, generate)).map(
      ([typeId, widgetState]) => [typeId, cloneQueueWidgetState(widgetState, typeId as WidgetTypeId)]
    )
  ) as WidgetStateMap;
  const graphHistorySnapshot = createGraphHistorySnapshot(`Queue snapshot ${queueItemId}`, graph);
  const generateSettings =
    route.sourceId === 'generate' ? normalizeGenerateSettings(widgetStates.generate.values) : null;
  const upscaleSettings =
    route.sourceId === 'upscale' ? normalizeUpscaleWidgetValues(widgetStates.upscale.values) : null;
  const backendGraph = graph.backendGraph;
  const canvasGenerateSettings = route.sourceId === 'canvas' ? normalizeGenerateSettings(generate?.values) : null;
  const sourceGenerateSettings =
    route.sourceId === 'canvas'
      ? canvasGenerateSettings
      : route.sourceId === 'generate'
        ? generateSettings
        : route.sourceId === 'upscale'
          ? upscaleSettings
          : null;
  // The prompts that actually generate: the authored text wrapped by the active
  // prompt template. Computed once here because this is the only place every
  // Generate-shaped route converges — `generate` and `canvas` both land here, as
  // do the topbar, hotkey and graph-preview submits. Upscale carries no template
  // and merges to identity.
  const effectivePrompts = sourceGenerateSettings ? getEffectivePrompts(sourceGenerateSettings) : null;
  // Dynamic prompting is a Generate setting, so only the routes compiled from
  // GenerateSettings honour it; Upscale keeps its prompt literal. The caller has
  // already expanded, so the queue item records the exact prompts it will submit.
  //
  // A one-prompt expansion counts: `a {red} cat` and a random sample of one both
  // resolve to a single concrete prompt, and dropping it here would fall the
  // submission back to the authored text — sending the literal `{…}` to the model.
  //
  // The gate reads the *merged* prompt: a template may introduce `{a|b}` that the
  // authored prompt never had, and the caller expanded the merged text too.
  const expandedPositivePrompts =
    route.sourceId !== 'upscale' &&
    compiled.positivePrompts &&
    compiled.positivePrompts.length > 0 &&
    effectivePrompts &&
    hasDynamicPromptSyntax(effectivePrompts.positivePrompt)
      ? compiled.positivePrompts
      : undefined;
  const expandedSeedBehaviour = expandedPositivePrompts
    ? (canvasGenerateSettings ?? generateSettings)?.dynamicPromptsSeedBehaviour
    : undefined;
  const backendSubmission: QueueCompiledSubmission = !backendGraph
    ? { error: `${route.sourceId} queue item is missing a compiled backend graph.`, kind: 'invalid' }
    : route.sourceId === 'workflow'
      ? {
          batchCount: sanitizeBatchCount(widgetStates.generate?.values.batchCount),
          graph: backendGraph,
          kind: 'workflow',
        }
      : sourceGenerateSettings && effectivePrompts
        ? {
            batchCount: sourceGenerateSettings.batchCount,
            graph: backendGraph,
            kind: 'generate',
            // A disabled negative prompt stays empty, which also suppresses the
            // template's negative side — switching the field off must not let a
            // template put one back.
            negativePrompt: sourceGenerateSettings.negativePromptEnabled ? effectivePrompts.negativePrompt : '',
            negativePromptNodeId: generate?.negativePromptNodeId ?? 'negative_prompt',
            positivePrompt: effectivePrompts.positivePrompt,
            positivePromptNodeId: generate?.positivePromptNodeId ?? 'positive_prompt',
            ...(expandedPositivePrompts ? { positivePrompts: expandedPositivePrompts } : {}),
            seed: sourceGenerateSettings.seed,
            ...(expandedSeedBehaviour ? { seedBehaviour: expandedSeedBehaviour } : {}),
            seedNodeId: generate?.seedNodeId ?? 'seed',
            shouldRandomizeSeed: sourceGenerateSettings.shouldRandomizeSeed,
          }
        : { error: `${route.sourceId} queue item is missing source submission metadata.`, kind: 'invalid' };
  const selectedGalleryBoardId = widgetStates.gallery?.values.selectedBoardId;
  const generatePresentationSettings = normalizeGenerateSettings(widgetStates.generate?.values);
  const presentationDimensions =
    route.sourceId === 'upscale' && upscaleSettings?.inputImage
      ? getUpscaleOutputDimensions(upscaleSettings.inputImage, upscaleSettings.scale)
      : {
          height: generatePresentationSettings?.height ?? project.canvas.document.height,
          width: generatePresentationSettings?.width ?? project.canvas.document.width,
        };
  const queueItem: QueueItem = {
    cancellable: backendSupportsCancellation,
    id: queueItemId,
    snapshot: {
      backendSubmission,
      canvas: canvasSnapshot ? structuredClone(canvasSnapshot) : cloneCanvas(project.canvas),
      destination: route.destination,
      filterIntermediateResults: route.sourceId === 'workflow',
      galleryBoardId: typeof selectedGalleryBoardId === 'string' ? selectedGalleryBoardId : null,
      ...(generate ? { generate: cloneQueueGenerateSnapshot(generate) } : {}),
      graph,
      presentation: {
        // Placeholder sizing only: superseded by the backend's real item ids as
        // soon as the batch is accepted.
        batchCount:
          backendSubmission.kind === 'invalid'
            ? 1
            : backendSubmission.batchCount * (expandedPositivePrompts?.length ?? 1),
        height: presentationDimensions.height,
        // The merged prompt, so the queue row reads the same before and after the
        // backend session arrives with its own (already merged) field values.
        ...(effectivePrompts?.positivePrompt ? { positivePrompt: effectivePrompts.positivePrompt } : {}),
        width: presentationDimensions.width,
      },
      sourceId: route.sourceId,
      ...(route.sourceId === 'generate' || route.sourceId === 'canvas'
        ? { resultNodeIds: ['canvas_output'] }
        : route.sourceId === 'upscale'
          ? { resultNodeIds: ['upscale_output'] }
          : {}),
      submittedAt,
      widgetInstances: cloneQueueWidgetInstances(project.widgetInstances),
      widgetStates,
    },
    status: 'pending',
  };

  return {
    ...project,
    events: [
      {
        createdAt: submittedAt,
        id: createId('event'),
        runId: queueItemId,
        summary: `Submitted immutable ${route.sourceId} graph snapshot to ${route.destination}`,
        type: 'queue-submitted',
      },
      ...project.events,
    ],
    graphHistory: prependGraphHistory(project.graphHistory, graphHistorySnapshot),
    promptHistory: generateSettings
      ? addPromptHistoryItem(project.promptHistory, getPromptHistoryItemFromGenerateSettings(generateSettings))
      : upscaleSettings
        ? addPromptHistoryItem(project.promptHistory, {
            negativePrompt: upscaleSettings.negativePromptEnabled ? upscaleSettings.negativePrompt : null,
            positivePrompt: upscaleSettings.positivePrompt,
          })
        : project.promptHistory,
    invocation: {
      ...project.invocation,
      destination: route.destination,
      lastSubmittedRunId: queueItemId,
      sourceId: route.sourceId,
    },
    queue: { items: [queueItem, ...project.queue.items] },
    widgetGraphs:
      route.sourceId === 'generate' || route.sourceId === 'upscale'
        ? { ...project.widgetGraphs, [route.sourceId]: cloneGraph(graph) }
        : project.widgetGraphs,
  };
};

const submitInvocationSnapshot = (
  project: Project,
  backendSupportsCancellation: boolean,
  route = resolveInvocationRoute(project),
  models?: readonly ModelConfig[],
  positivePrompts?: string[]
): Project => {
  if (!isInvocationRouteValid(route)) {
    return project;
  }

  const compiledSnapshot = compileInvocationSnapshot(project, route, models);

  if (!compiledSnapshot) {
    return project;
  }

  return enqueueCompiledSnapshot(project, route, { ...compiledSnapshot, positivePrompts }, backendSupportsCancellation);
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
      // Hydrated from the library (Open dialog or a deep link). Opening an
      // already-open project just focuses its tab.
      if (state.projects.some((project) => project.id === action.project.id)) {
        return { ...state, activeProjectId: action.project.id };
      }

      const project = normalizeWorkbenchProject(action.project);

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
      const nextState = updateActiveProjectLayoutPreset(state, preset, { applyDefaultRoute: true });

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
        snapshot: createLayoutPresetSnapshot(normalizeWorkbenchProject(activeProject)),
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

      return {
        ...state,
        account: { ...state.account, activeLayoutPresetId: preset.id, customLayoutPresets, layoutPresetOrder },
      };
    }
    case 'saveLayoutPreset': {
      const activeProject = state.projects.find((project) => project.id === state.activeProjectId);

      if (!activeProject) {
        return state;
      }

      const snapshot = createLayoutPresetSnapshot(normalizeWorkbenchProject(activeProject));

      // Built-in preset bodies are code, so their saved form lives in an
      // override map; custom presets own their snapshot outright.
      if (isBuiltInLayoutPresetId(action.presetId)) {
        const layoutPresetOverrides: LayoutPresetOverrides = {
          ...state.account.layoutPresetOverrides,
          [action.presetId]: { ...snapshot, layout: { ...snapshot.layout, presetId: action.presetId } },
        };

        return { ...state, account: { ...state.account, layoutPresetOverrides } };
      }

      const customLayoutPresets = (state.account.customLayoutPresets ?? []).map((preset) =>
        preset.id === action.presetId
          ? { ...preset, snapshot: { ...snapshot, layout: { ...snapshot.layout, presetId: preset.id } } }
          : preset
      );

      return { ...state, account: { ...state.account, customLayoutPresets } };
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
      const projects = state.projects.map((project) =>
        project.layout.presetId === action.presetId
          ? { ...project, layout: { ...project.layout, presetId: defaultLayoutPreset.id } }
          : project
      );

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
      const preset = getAvailableLayoutPreset(
        state,
        state.projects.find((project) => project.id === state.activeProjectId)?.layout.presetId ??
          state.account.activeLayoutPresetId
      );

      return updateActiveProjectLayoutPreset(state, preset, { applyDefaultRoute: false });
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
        const instanceIds = region.instanceIds.includes(instanceId)
          ? region.instanceIds
          : [...region.instanceIds, instanceId];
        const widgetInstances = project.widgetInstances[instanceId]
          ? project.widgetInstances
          : {
              ...project.widgetInstances,
              [instanceId]: createWidgetInstance(action.widgetId, instanceId, action.initialValues),
            };
        // Placing an instance into a region implicitly docks it: an instance
        // must never render in a panel and a floating window at once.
        const { [instanceId]: _floated, ...floatingWidgets } = project.floatingWidgets ?? {};

        return {
          ...project,
          floatingWidgets,
          layout: openPanelForRegion(project.layout, action.region),
          widgetInstances,
          widgetRegions: {
            ...project.widgetRegions,
            [action.region]: {
              ...region,
              activeInstanceId: instanceId,
              instanceIds,
              isCollapsed: false,
            },
          },
        };
      });
    }
    case 'selectRegionWidget': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) => {
        const region = project.widgetRegions[action.region];

        if (action.region === 'center') {
          return {
            ...project,
            widgetRegions: {
              ...project.widgetRegions,
              center: { ...region, activeInstanceId: action.widgetId, isCollapsed: false },
            },
          };
        }

        const widgetRegion =
          region.activeInstanceId === action.widgetId
            ? { ...region, isCollapsed: !region.isCollapsed }
            : { ...region, activeInstanceId: action.widgetId, isCollapsed: false };

        return {
          ...project,
          layout: openPanelForRegion(project.layout, action.region),
          widgetRegions: { ...project.widgetRegions, [action.region]: widgetRegion },
        };
      });
    }
    case 'toggleRegionWidget': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) =>
        updateProjectWidgetRegion(project, action.region, (region) => {
          const isEnabled = region.instanceIds.includes(action.widgetId);

          if (action.region === 'center' && isEnabled && region.instanceIds.length === 1) {
            return region;
          }

          const instanceIds = isEnabled
            ? region.instanceIds.filter((widgetId) => widgetId !== action.widgetId)
            : [...region.instanceIds, action.widgetId];
          const fallbackInstanceId = getNextInstanceId(region, action.widgetId);

          return {
            ...region,
            activeInstanceId: isEnabled && fallbackInstanceId ? fallbackInstanceId : action.widgetId,
            instanceIds,
            isCollapsed: action.region === 'center' ? false : instanceIds.length === 0 ? true : region.isCollapsed,
          };
        })
      );
    }
    case 'floatWidget': {
      return updateActiveProject(state, (project) => {
        if (project.floatingWidgets?.[action.instanceId]) {
          return project;
        }

        const hostEntry = (Object.entries(project.widgetRegions) as [WidgetRegion, WidgetRegionState][]).find(
          ([, region]) => region.instanceIds.includes(action.instanceId)
        );

        if (!hostEntry || !project.widgetInstances[action.instanceId]) {
          return project;
        }

        const [hostRegionId, hostRegion] = hostEntry;

        // The work surface must keep a view. `toggleRegionWidget` and
        // `closeWidgetPlacement` refuse the same removal; floating it out is
        // the same removal with a window attached.
        if (hostRegionId === 'center' && hostRegion.instanceIds.length === 1) {
          return project;
        }

        const instanceIds = hostRegion.instanceIds.filter((instanceId) => instanceId !== action.instanceId);
        const fallbackInstanceId = getNextInstanceId(hostRegion, action.instanceId);
        const floating: FloatingWidgetState = {
          ...cascadeDefaultGeometry(Object.keys(project.floatingWidgets ?? {}).length),
          mode: 'windowed',
          returnIndex: hostRegion.instanceIds.indexOf(action.instanceId),
          returnRegion: hostRegionId,
          stackOrder: nextStackOrder(project.floatingWidgets),
        };

        return {
          ...project,
          floatingWidgets: { ...project.floatingWidgets, [action.instanceId]: floating },
          widgetRegions: {
            ...project.widgetRegions,
            [hostRegionId]: {
              ...hostRegion,
              activeInstanceId:
                hostRegion.activeInstanceId === action.instanceId && fallbackInstanceId
                  ? fallbackInstanceId
                  : hostRegion.activeInstanceId,
              instanceIds,
              // Floating the last widget out of a rail leaves nothing to show,
              // so the rail collapses rather than standing open and empty —
              // the same repair `toggleRegionWidget` makes.
              isCollapsed: instanceIds.length === 0 ? true : hostRegion.isCollapsed,
            },
          },
        };
      });
    }
    case 'dockFloatingWidget': {
      return updateActiveProject(state, (project) => {
        const floating = project.floatingWidgets?.[action.instanceId];

        if (!floating) {
          return project;
        }

        const { [action.instanceId]: _docked, ...remaining } = project.floatingWidgets ?? {};
        const region = project.widgetRegions[floating.returnRegion];
        const instanceIds = region.instanceIds.includes(action.instanceId)
          ? region.instanceIds
          : insertAtReturnIndex(region.instanceIds, action.instanceId, floating.returnIndex);

        return {
          ...project,
          floatingWidgets: remaining,
          layout: openPanelForRegion(project.layout, floating.returnRegion),
          widgetRegions: {
            ...project.widgetRegions,
            [floating.returnRegion]: {
              ...region,
              activeInstanceId: action.instanceId,
              instanceIds,
              isCollapsed: false,
            },
          },
        };
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

        // A pointer-down/up on the title bar with no movement still commits the
        // starting geometry. Without this the project is marked dirty — and
        // autosaved — every time someone clicks the window chrome.
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
    case 'focusFloatingWidget': {
      return updateActiveProject(state, (project) => {
        const floating = project.floatingWidgets?.[action.instanceId];
        const topOrder = nextStackOrder(project.floatingWidgets) - 1;

        if (!floating || floating.stackOrder === topOrder) {
          return project;
        }

        // Renumbered to a compact 1..N rather than appended above the current
        // top. `stackOrder` is persisted, so a counter that only ever climbs
        // writes ever-larger numbers into the document for what is really a
        // reordering of the same few windows.
        const below = Object.entries(project.floatingWidgets ?? {})
          .filter(([instanceId]) => instanceId !== action.instanceId)
          .sort(([, left], [, right]) => left.stackOrder - right.stackOrder);
        const floatingWidgets: Record<WidgetInstanceId, FloatingWidgetState> = {};

        for (const [instanceId, windowState] of below) {
          floatingWidgets[instanceId] = { ...windowState, stackOrder: Object.keys(floatingWidgets).length + 1 };
        }

        floatingWidgets[action.instanceId] = { ...floating, stackOrder: below.length + 1 };

        return { ...project, floatingWidgets };
      });
    }
    case 'moveWidgetInstance': {
      return updateActiveProject(state, (project) => {
        const fromRegion = project.widgetRegions[action.fromRegion];
        const toRegion = project.widgetRegions[action.toRegion];
        const nextFromInstanceIds = fromRegion.instanceIds.filter((instanceId) => instanceId !== action.instanceId);
        const nextToInstanceIds = insertAt(toRegion.instanceIds, action.instanceId, action.toIndex);

        return {
          ...project,
          layout: openPanelForRegion(project.layout, action.toRegion),
          widgetRegions: {
            ...project.widgetRegions,
            [action.fromRegion]: {
              ...fromRegion,
              activeInstanceId:
                fromRegion.activeInstanceId === action.instanceId
                  ? (nextFromInstanceIds[0] ?? fromRegion.activeInstanceId)
                  : fromRegion.activeInstanceId,
              instanceIds: nextFromInstanceIds,
              isCollapsed:
                action.fromRegion === 'center' ? false : nextFromInstanceIds.length === 0 || fromRegion.isCollapsed,
            },
            [action.toRegion]: {
              ...toRegion,
              activeInstanceId: action.instanceId,
              instanceIds: nextToInstanceIds,
              isCollapsed: false,
            },
          },
        };
      });
    }
    case 'reorderWidgetInstances': {
      return updateActiveWidgetRegion(state, action.region, (region) => ({
        ...region,
        activeInstanceId: action.activeInstanceId ?? region.activeInstanceId,
        instanceIds: action.instanceIds,
      }));
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
          : applyAutoRouteForEdit(updated, 'upscale', context);
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
    case 'applyProjectGraphAction': {
      return updateActiveProject(state, (project) => {
        const projectGraph = projectGraphReducer(project.projectGraph, action.action);

        if (projectGraph === project.projectGraph) {
          return project;
        }

        const routedProject = isHighConfidenceGraphEdit(action.action)
          ? applyAutoRouteForEdit(project, 'workflow', context)
          : project;
        const undoLabel = getProjectGraphUndoLabel(action.action);
        const nextProject = undoLabel ? pushUndo(routedProject, undoLabel) : routedProject;
        const updated = { ...nextProject, projectGraph };

        return updated;
      });
    }
    case 'replaceProjectGraph': {
      let didRetainOutgoingGraph = true;
      const nextState = updateActiveProject(state, (project) => {
        // One immutable clone is shared by undo and graph history; neither
        // snapshot path mutates the document.
        const routedProject = applyAutoRouteForEdit(project, 'workflow', context);
        const outgoingGraph = cloneProjectGraph(project.projectGraph);
        const nextProject = pushUndo(routedProject, 'Replace project graph', outgoingGraph);
        const historySnapshot = createDocumentHistorySnapshot(`Before: ${action.label}`, outgoingGraph, false);
        didRetainOutgoingGraph = (historySnapshot.retainedBytes ?? 0) <= GRAPH_HISTORY_BYTE_BUDGET;

        return {
          ...nextProject,
          events: [
            {
              createdAt: now(),
              id: createId('event'),
              summary: `Replaced the project graph with "${action.document.name || 'Untitled Workflow'}" (${action.label})`,
              type: 'graph-replaced',
            },
            ...nextProject.events,
          ],
          graphHistory: prependGraphHistory(nextProject.graphHistory, historySnapshot),
          projectGraph: cloneProjectGraph(action.document),
        };
      });
      const activeProject = nextState.projects.find((project) => project.id === nextState.activeProjectId);

      return addNotification(
        nextState,
        createNotification({
          kind: 'info',
          message: didRetainOutgoingGraph
            ? 'The previous project graph was saved to graph history.'
            : 'The previous project graph exceeded the 64 MiB history budget, so its graph-history snapshot was skipped. Undo remains available for this session.',
          projectId: activeProject?.id,
          title: `Project graph replaced (${action.label})`,
        })
      );
    }
    case 'saveProjectGraphSnapshot': {
      return updateActiveProject(state, (project) => ({
        ...project,
        events: [
          {
            createdAt: now(),
            id: createId('event'),
            summary: `Saved a graph history snapshot of "${project.projectGraph.name || 'Untitled Workflow'}"`,
            type: 'graph-snapshot-saved',
          },
          ...project.events,
        ],
        graphHistory: prependGraphHistory(
          project.graphHistory,
          createDocumentHistorySnapshot(
            `Manual save: ${project.projectGraph.name || 'Untitled Workflow'}`,
            project.projectGraph
          )
        ),
      }));
    }
    case 'restoreProjectGraphSnapshot': {
      return updateActiveProject(state, (project) => {
        const snapshot = project.graphHistory.find((entry) => entry.id === action.snapshotId);

        if (!snapshot?.document) {
          return project;
        }

        const routedProject = applyAutoRouteForEdit(project, 'workflow', context);
        const nextProject = pushUndo(routedProject, 'Restore graph history snapshot');

        return {
          ...nextProject,
          graphHistory: prependGraphHistory(
            nextProject.graphHistory,
            createDocumentHistorySnapshot('Before restore', project.projectGraph)
          ),
          projectGraph: cloneProjectGraph(normalizeProjectGraph(snapshot.document)),
        };
      });
    }
    case 'setProjectGraphLibraryBinding': {
      return updateActiveProject(state, (project) => ({
        ...project,
        projectGraph: { ...project.projectGraph, libraryWorkflowId: action.libraryWorkflowId },
      }));
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
      return withEnqueueNotification(
        state,
        updateActiveProject(state, (project) =>
          submitInvocationSnapshot(
            project,
            action.backendSupportsCancellation,
            resolveInvocationRoute(project, 'global', action.route, action.models),
            action.models,
            action.positivePrompts
          )
        ),
        state.activeProjectId
      );
    }
    case 'markQueueItemBackendSubmitted': {
      return updateProjectById(state, action.projectId, (project) =>
        clampCanvasStagingSelection(
          updateQueueItem(project, action.queueItemId, (item) => {
            const status = item.status === 'cancelled' ? 'cancelled' : 'running';
            const hasSameBackendItemIds =
              item.backendItemIds?.length === action.backendItemIds.length &&
              item.backendItemIds.every((id, index) => id === action.backendItemIds[index]);

            return item.backendBatchId === action.backendBatchId && hasSameBackendItemIds && item.status === status
              ? item
              : { ...item, backendBatchId: action.backendBatchId, backendItemIds: action.backendItemIds, status };
          })
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

      const nextState = updateProjectById(state, action.projectId, (project) =>
        clampCanvasStagingSelection(
          updateQueueItem(project, action.queueItemId, (item) => ({
            ...item,
            error: action.error,
            status: action.status,
          }))
        )
      );

      if (action.notify === false || (action.status !== 'failed' && action.status !== 'cancelled')) {
        return nextState;
      }

      return addNotification(
        nextState,
        createNotification({
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
        routeQueueItemPartialResults(project, action.queueItemId, action.backendItemId, action.images)
      );
    }
    case 'markQueueItemBackendCancelled': {
      const project = state.projects.find((project) => project.id === action.projectId);
      const queueItem = project?.queue.items.find((item) => item.id === action.queueItemId);

      if (queueItem?.status === 'cancelled' || queueItem?.status === 'completed') {
        return state;
      }

      return updateProjectById(state, action.projectId, (project) => {
        const nextProject = updateQueueItem(project, action.queueItemId, (item) => {
          const cancelledBackendItemIds = mergeBackendItemId(item.cancelledBackendItemIds, action.backendItemId);

          return {
            ...item,
            cancelledBackendItemIds,
            status: getQueueItemStatusAfterBackendCancellation(item, cancelledBackendItemIds),
          };
        });

        return clampCanvasStagingSelection(nextProject);
      });
    }
    case 'routeQueueItemResults': {
      const project = state.projects.find((project) => project.id === action.projectId);
      const queueItem = project?.queue.items.find((item) => item.id === action.queueItemId);

      if (queueItem?.status === 'cancelled') {
        return state;
      }

      const nextState = updateProjectById(state, action.projectId, (project) =>
        routeQueueItemResults(project, action.queueItemId, action.images)
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
          const existingNavigationQuery =
            values.selectedImageQuery && typeof values.selectedImageQuery === 'object'
              ? (values.selectedImageQuery as Record<string, unknown>)
              : null;
          const selectedImageQuery =
            action.preserveNavigationQuery && existingNavigationQuery
              ? { ...existingNavigationQuery, page: selectedImagePage }
              : {
                  boardId: typeof values.selectedBoardId === 'string' ? values.selectedBoardId : 'none',
                  galleryView: values.galleryView === 'assets' ? 'assets' : 'images',
                  imageOrderDir: settings.imageOrderDir,
                  page: selectedImagePage,
                  paginationMode: settings.paginationMode,
                  searchTerm: typeof values.searchTerm === 'string' ? values.searchTerm : '',
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
                boardId: typeof values.selectedBoardId === 'string' ? values.selectedBoardId : 'none',
                galleryView: values.galleryView === 'assets' ? 'assets' : 'images',
                imageOrderDir: settings.imageOrderDir,
                page: selectedImagePage,
                paginationMode: settings.paginationMode,
                searchTerm: typeof values.searchTerm === 'string' ? values.searchTerm : '',
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
          const selectedImagePage =
            typeof values.galleryPage === 'number' && Number.isFinite(values.galleryPage)
              ? Math.max(0, Math.floor(values.galleryPage))
              : 0;

          return {
            ...values,
            ...(action.primaryItem.kind === 'video' ? { compareImage: null } : {}),
            selectedImage: action.primaryItem,
            selectedImageName: toGalleryItemKey(action.primaryItem),
            selectedImageNames: action.itemKeys,
            selectedImagePage,
            selectedImageQuery: {
              boardId: typeof values.selectedBoardId === 'string' ? values.selectedBoardId : 'none',
              galleryView: values.galleryView === 'assets' ? 'assets' : 'images',
              imageOrderDir: settings.imageOrderDir,
              page: selectedImagePage,
              paginationMode: settings.paginationMode,
              searchTerm: typeof values.searchTerm === 'string' ? values.searchTerm : '',
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
      return updateGalleryValues(
        state,
        (values) => ({ ...values, galleryPage: Math.max(0, action.page) }),
        action.projectId
      );
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
      const nextState = updateProjectById(state, targetProjectId, (project) =>
        clampCanvasStagingSelection({
          ...project,
          queue: {
            items: project.queue.items.map((item) => {
              if (item.id !== action.queueItemId || !isCancellableQueueItem(item)) {
                return item;
              }

              return { ...item, status: 'cancelled' };
            }),
          },
        })
      );

      if (!targetProject || !queueItem || !canCancelQueueItem) {
        return nextState;
      }

      return addNotification(
        nextState,
        createNotification({
          kind: 'info',
          message: `${targetProject.name}: ${action.queueItemId}`,
          projectId: targetProject.id,
          title: 'Invocation cancellation requested',
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
        projects: state.projects.map((project) =>
          clampCanvasStagingSelection({
            ...project,
            queue: {
              items: shouldApplyQueueBulkActionToProject(project, action.projectId)
                ? project.queue.items.map((item) =>
                    isCancellableQueueItem(item) ? { ...item, status: 'cancelled' } : item
                  )
                : project.queue.items,
            },
          })
        ),
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
        projects: state.projects.map((project) =>
          clampCanvasStagingSelection({
            ...project,
            queue: {
              items: shouldApplyQueueBulkActionToProject(project, action.projectId)
                ? project.queue.items.map((item) =>
                    isCancellableQueueItem(item) && item.id !== action.currentQueueItemId
                      ? { ...item, status: 'cancelled' }
                      : item
                  )
                : project.queue.items,
            },
          })
        ),
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
        projects: state.projects.map((project) => ({
          ...project,
          queue: { items: project.queue.items.filter((item) => !isClearableQueueItem(item)) },
        })),
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
          graphHistory: project.graphHistory,
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
          graphHistory: project.graphHistory,
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
    case 'reconcileProjectConflict': {
      // A save lost the revision race against another tab/device. The server
      // version takes over the original project id, and the local edits
      // continue in the recovered fork — which stays the active project when
      // the user was looking at it.
      const normalizedServerProject = normalizeWorkbenchProject(action.serverProject);
      const localProject = state.projects.find((project) => project.id === action.projectId);
      const hasOriginal = localProject !== undefined;
      const recoveredProject = recoverProjectUnderNewIdentity(
        localProject,
        action.recoveredProject,
        action.recoveredIdentity
      );
      // The server document replaces the local one under the SAME project id, so a
      // live engine mirroring that id may hold pixel history for the outgoing
      // document. Bump the revision past both sides so the mirror treats the swap
      // as a document replacement (clearing that history) even when dims/layer ids
      // coincide. (The recovered fork gets a fresh project id → a fresh engine.)
      const serverProject: Project = hasOriginal
        ? {
            ...normalizedServerProject,
            canvas: {
              ...normalizedServerProject.canvas,
              documentRevision:
                Math.max(normalizedServerProject.canvas.documentRevision, localProject.canvas.documentRevision) + 1,
            },
          }
        : normalizedServerProject;
      const projects = hasOriginal
        ? state.projects.flatMap((project) =>
            project.id === action.projectId ? [serverProject, recoveredProject] : [project]
          )
        : [...state.projects, serverProject, recoveredProject];

      return addNotification(
        {
          ...state,
          activeProjectId: state.activeProjectId === action.projectId ? recoveredProject.id : state.activeProjectId,
          projects,
        },
        createNotification({
          kind: 'info',
          message: `"${serverProject.name}" was changed elsewhere. Your local edits continue in "${recoveredProject.name}" — manage recoveries in the Project panel.`,
          title: 'Project recovered',
        })
      );
    }
    case 'reconcileDeletedProject': {
      // The project was deleted on another device while this one held unsaved edits. Unlike a
      // revision conflict there is no server version to adopt — the deletion is the server's
      // answer. Re-creating the id would undo it everywhere, so the local edits continue under a
      // fresh identity and the original simply goes.
      const localProject = state.projects.find((project) => project.id === action.projectId);
      const hasOriginal = localProject !== undefined;
      const recoveredProject = recoverProjectUnderNewIdentity(
        localProject,
        action.recoveredProject,
        action.recoveredIdentity
      );
      const projects = hasOriginal
        ? state.projects.map((project) => (project.id === action.projectId ? recoveredProject : project))
        : [...state.projects, recoveredProject];

      return addNotification(
        {
          ...state,
          activeProjectId: state.activeProjectId === action.projectId ? recoveredProject.id : state.activeProjectId,
          projects,
        },
        createNotification({
          kind: 'info',
          message: `That project was deleted elsewhere. Your edits — including anything typed since — continue in "${recoveredProject.name}".`,
          title: 'Project recovered',
        })
      );
    }
    case 'autosaveStarted': {
      return { ...state, autosave: { status: 'saving' } };
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
      const detail = action.context?.error;
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

        return Object.entries(settings).every(([key, value]) => {
          const settingKey = key as keyof ProjectSettings;

          return areProjectSettingValuesEqual(
            project.settings[settingKey],
            value as ProjectSettings[typeof settingKey]
          );
        })
          ? project
          : { ...project, settings };
      });
    }
  }
};

export type __WorkbenchReducerActionInternal = WorkbenchReducerAction;
