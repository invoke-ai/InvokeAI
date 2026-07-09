import type { GallerySettings } from './gallery/settings';
import type { GenerateWidgetValues } from './generation/types';
import type { ModelConfig } from './models/types';
import type {
  CanvasAdjustmentsContract,
  CanvasControlAdapterContract,
  CanvasControlLayerContract,
  CanvasDocumentContractV2,
  CanvasLayerBaseContract,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasMaskContract,
  CanvasPlacementContract,
  CanvasRegionalGuidanceLayerContract,
  CanvasStagingAreaContractV2,
  CanvasStateContractV2,
  CenterViewId,
  CanvasRasterLayerContractV2,
  CanvasStagingCandidateContract,
  DeveloperLogNamespace,
  GeneratedImageContract,
  GraphContract,
  GraphHistorySnapshot,
  InvocationRoute,
  InvocationSourceId,
  LayoutPreset,
  LayoutPresetId,
  LayoutPresetSnapshot,
  Project,
  ProjectLayoutState,
  ProjectSettings,
  ProjectUndoSnapshot,
  PromptHistoryItem,
  QueueItem,
  QueueItemStatus,
  ResultDestination,
  WidgetFailure,
  WidgetId,
  WidgetInstanceContract,
  WidgetInstanceId,
  WidgetRegion,
  WidgetRegionState,
  WidgetStateContract,
  WidgetStateMap,
  WidgetTypeId,
  WorkbenchNotification,
  WorkbenchNotificationKind,
  WorkbenchState,
} from './types';
import type { ProjectGraphState } from './workflows/types';

import { createNewCanvasStateV2, migrateCanvasStateToV2 } from './canvasMigration';
import { getGenerationModelAvailabilityReasons } from './generation/baseGenerationPolicies';
import { sanitizeBatchCount } from './generation/batch';
import { compileGenerateGraph, resolveGenerateSeed } from './generation/graph';
import {
  addPromptHistoryItem,
  getPromptHistoryItemFromGenerateSettings,
  removePromptHistoryItem,
} from './generation/promptHistory';
import {
  cloneGenerateWidgetValues,
  normalizeGenerateSettings,
  normalizeGenerateWidgetValues,
  syncGenerateWidgetValuesWithModels,
} from './generation/settings';
import {
  defaultInvocationRoute,
  isInvocationRouteValid,
  isInvocationSourceAvailable,
  resolveInvocationRoute,
} from './invocation';
import { defaultLayoutPreset, getLayoutPreset } from './layoutPresets';
import { cloneLayoutPresetWidgetRegions, createLayoutPresetSnapshot } from './layoutPresetSnapshots';
import { normalizeProjectSettings } from './settings/store';
import { compileProjectGraph } from './workflows/buildGraph';
import {
  cloneProjectGraph,
  createProjectGraph,
  getProjectGraphUndoLabel,
  isHighConfidenceGraphEdit,
  normalizeProjectGraph,
  projectGraphReducer,
  type ProjectGraphAction,
} from './workflows/document';
import { getInvocationTemplatesSnapshot } from './workflows/templates';

/** Partial of the base props shared by every canvas layer. `transform` merges field-wise. */
type CanvasLayerBasePatch = Partial<
  Pick<CanvasLayerBaseContract, 'name' | 'isEnabled' | 'isLocked' | 'opacity' | 'blendMode'>
> & { transform?: Partial<CanvasLayerBaseContract['transform']> };

/** Per-layer-type config patch for `updateCanvasLayerConfig`; a mismatched `layerType` is ignored. */
type CanvasLayerConfigPatch =
  | { layerType: 'raster'; adjustments?: CanvasAdjustmentsContract; isTransparencyLocked?: boolean }
  | {
      layerType: 'control';
      adapter?: Partial<CanvasControlAdapterContract>;
      withTransparencyEffect?: boolean;
      filter?: CanvasControlLayerContract['filter'];
    }
  | {
      layerType: 'regional_guidance';
      mask?: Partial<CanvasMaskContract>;
      positivePrompt?: string | null;
      negativePrompt?: string | null;
      autoNegative?: boolean;
      referenceImages?: CanvasRegionalGuidanceLayerContract['referenceImages'];
    }
  | { layerType: 'inpaint_mask'; mask?: Partial<CanvasMaskContract>; noiseLevel?: number; denoiseLimit?: number };

type QueueGenerateSnapshot = NonNullable<QueueItem['snapshot']['generate']>;

type WorkbenchAction =
  | { type: 'createProject' }
  | { type: 'openProject'; project: Project }
  | { type: 'closeProject'; projectId: string }
  | { type: 'renameProject'; projectId: string; name: string }
  | { type: 'switchProject'; projectId: string }
  | { type: 'setCenterView'; centerViewId: CenterViewId }
  | { type: 'applyPreset'; presetId: LayoutPresetId }
  | { type: 'addLayoutPreset'; presetId: LayoutPresetId; label: string }
  | { type: 'renameLayoutPreset'; presetId: LayoutPresetId; label: string }
  | { type: 'deleteLayoutPreset'; presetId: LayoutPresetId }
  | { type: 'resetActiveLayout' }
  | { type: 'recoverShellLayout' }
  | { type: 'setInvocationSource'; sourceId: InvocationSourceId }
  | { type: 'setInvocationDestination'; destination: ResultDestination }
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
  | { type: 'setGenerateSettings'; values: GenerateWidgetValues; projectId?: string }
  | { type: 'patchGenerateSettings'; values: Partial<GenerateWidgetValues>; projectId?: string }
  | { type: 'setGenerateBatchCount'; batchCount: number; projectId?: string }
  | { type: 'addPromptToHistory'; prompt: PromptHistoryItem; projectId?: string }
  | { type: 'removePromptFromHistory'; prompt: PromptHistoryItem; projectId?: string }
  | { type: 'clearPromptHistory'; projectId?: string }
  | { type: 'patchWidgetValues'; widgetId: WidgetTypeId; values: Record<string, unknown>; projectId?: string }
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
      status: QueueItemStatus;
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
  | { type: 'setStagedImageIndex'; imageIndex: number }
  | { type: 'cycleStagedImage'; direction: -1 | 1 }
  | { type: 'discardSelectedStagedImage' }
  | { type: 'discardAllStagedImages' }
  | { type: 'toggleCanvasStagingVisibility' }
  | { type: 'toggleCanvasStagingThumbnailsVisibility' }
  | { type: 'selectGalleryImage'; image: GeneratedImageContract; projectId?: string }
  | { type: 'toggleGalleryImageInSelection'; image: GeneratedImageContract; projectId?: string }
  | { type: 'setGalleryMultiSelection'; imageNames: string[]; primaryImage: GeneratedImageContract; projectId?: string }
  | { type: 'setGalleryCompareImage'; image: GeneratedImageContract | null; projectId?: string }
  | { type: 'selectGalleryBoard'; boardId: string; projectId?: string }
  | { type: 'setGalleryView'; galleryView: 'images' | 'assets'; projectId?: string }
  | { type: 'setGallerySearchTerm'; searchTerm: string; projectId?: string }
  | { type: 'updateGallerySettings'; settings: Partial<GallerySettings>; projectId?: string }
  | { type: 'setGalleryPage'; page: number; projectId?: string }
  | { type: 'setGalleryPageInfo'; totalImages: number; projectId?: string }
  | { type: 'touchGalleryRefresh'; projectId?: string }
  | { type: 'touchGalleryImagesRefresh'; projectId?: string }
  | { type: 'removeGalleryImages'; imageNames: string[]; projectId?: string }
  | { type: 'setGalleryProjectBoardId'; boardId: string; projectId?: string }
  | { type: 'acceptStagedImage' }
  | { type: 'clearCanvasStaging' }
  | { type: 'addCanvasLayer'; layer: CanvasLayerContract; index?: number }
  | { type: 'removeCanvasLayers'; ids: string[] }
  | { type: 'duplicateCanvasLayer'; sourceId: string; newId: string }
  | { type: 'reorderCanvasLayers'; orderedIds: string[] }
  | { type: 'updateCanvasLayer'; id: string; patch: CanvasLayerBasePatch }
  | { type: 'setCanvasLayersEnabled'; updates: readonly { id: string; isEnabled: boolean }[] }
  | { type: 'updateCanvasLayerSource'; id: string; source: CanvasLayerSourceContract }
  | { type: 'updateCanvasLayerConfig'; id: string; config: CanvasLayerConfigPatch }
  | { type: 'convertCanvasLayer'; id: string; targetType: CanvasLayerContract['type']; layer: CanvasLayerContract }
  | {
      type: 'mergeCanvasLayersDown';
      upperLayerId: string;
      source: Extract<CanvasLayerSourceContract, { type: 'paint' }>;
    }
  | { type: 'setCanvasBbox'; bbox: CanvasDocumentContractV2['bbox'] }
  | { type: 'setCanvasSelectedLayer'; id: string | null }
  | { type: 'resizeCanvasDocument'; width: number; height: number; offsetX?: number; offsetY?: number }
  | { type: 'replaceCanvasDocument'; document: CanvasDocumentContractV2 }
  | { type: 'saveCanvasSnapshot'; id: string; name: string; createdAt: string }
  | { type: 'restoreCanvasSnapshot'; snapshotId: string }
  | { type: 'deleteCanvasSnapshot'; snapshotId: string }
  | { type: 'setCanvasStagingAutoSwitch'; mode: CanvasStagingAreaContractV2['autoSwitchMode'] }
  | {
      type: 'submitCanvasInvocationSnapshot';
      backendSupportsCancellation: boolean;
      destination: ResultDestination;
      generate: QueueGenerateSnapshot;
      graph: GraphContract;
      projectId: string;
    }
  | { type: 'cancelQueueItem'; queueItemId: string; projectId?: string }
  | { type: 'cancelAllQueueItems'; projectId?: string }
  | { type: 'cancelAllQueueItemsExceptCurrent'; projectId?: string; currentQueueItemId?: string | null }
  | { type: 'clearCompletedQueueItems' }
  | { type: 'undoProjectChange' }
  | { type: 'redoProjectChange' }
  | { type: 'hydrateWorkbench'; state: WorkbenchState }
  | { type: 'reconcileProjectConflict'; projectId: string; serverProject: Project; recoveredProject: Project }
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
      context?: Record<string, unknown>;
      namespace?: DeveloperLogNamespace;
      projectId?: string;
    }
  | { type: 'setBackendConnectionStatus'; status: WorkbenchState['backendConnection']['status']; error?: string }
  | { type: 'refreshBackendData' }
  | { type: 'recordNotice'; kind: WorkbenchNotificationKind; title: string; message?: string };

const HISTORY_LIMIT = 40;
const NOTIFICATION_LIMIT = 100;
const MIN_PANEL_SIZE_PX = 180;
const MAX_PANEL_SIZE_PX = 520;
const MIN_STATUS_PANEL_SIZE_PX = 96;
const MAX_STATUS_PANEL_SIZE_PX = 420;

const now = (): string => new Date().toISOString();

const createId = (prefix: string): string =>
  `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

const createNotification = ({
  kind,
  message,
  projectId,
  title,
}: {
  kind: WorkbenchNotificationKind;
  message?: string;
  projectId?: string;
  title: string;
}): WorkbenchNotification => ({
  createdAt: now(),
  id: createId('notification'),
  isRead: false,
  kind,
  message,
  projectId,
  title,
});

const addNotification = (state: WorkbenchState, notification: WorkbenchNotification): WorkbenchState => ({
  ...state,
  notifications: [notification, ...state.notifications].slice(0, NOTIFICATION_LIMIT),
});

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
  document: Pick<CanvasDocumentContractV2, 'height' | 'width'>
): CanvasStagingCandidateContract => ({
  ...image,
  placement:
    'placement' in image && image.placement
      ? clonePlacement(image.placement)
      : createCenteredPlacement(image, document),
});

const clearStagingArea = (stagingArea: CanvasStateContractV2['stagingArea']): CanvasStateContractV2['stagingArea'] => ({
  ...stagingArea,
  isVisible: false,
  pendingImageIds: [],
  pendingImages: [],
  selectedImageIndex: 0,
  sourceQueueItemId: undefined,
});

const clampStagedImageIndex = (imageIndex: number, pendingImageCount: number): number => {
  const maxIndex = Math.max(0, pendingImageCount - 1);

  return Math.min(maxIndex, Math.max(0, imageIndex));
};

const cycleStagedImageIndex = (imageIndex: number, pendingImageCount: number, direction: -1 | 1): number => {
  if (pendingImageCount < 2) {
    return 0;
  }

  return (imageIndex + direction + pendingImageCount) % pendingImageCount;
};

/**
 * Picks the staging selection after a batch of results arrives, honoring
 * `stagingArea.autoSwitchMode`. With nothing new, the current index is kept
 * (clamped). `off` preserves the prior behavior of focusing the first freshly
 * arrived candidate; `latest`/`oldest` jump to the newest/oldest pending image.
 */
const resolveStagingSelectionIndex = (
  mode: CanvasStagingAreaContractV2['autoSwitchMode'],
  currentIndex: number,
  existingCount: number,
  incomingCount: number,
  totalCount: number
): number => {
  if (incomingCount === 0) {
    return clampStagedImageIndex(currentIndex, totalCount);
  }

  switch (mode) {
    case 'latest':
      return Math.max(0, totalCount - 1);
    case 'oldest':
      return 0;
    default:
      return existingCount;
  }
};

// ---------------------------------------------------------------------------
// Canvas v2 layer reducers
//
// Dispatched one-per-gesture by the canvas engine and layers panel. Each mutates
// only `project.canvas` with a small serializable payload and never touches
// pixels or calls `pushUndo` — canvas history is owned by the rendering engine,
// not the project undo stack. Invariants upheld everywhere: layer index 0 is the
// top-most layer, `document.selectedLayerId` is always null or a present layer id,
// and untouched layers keep their object identity (reference-preserving updates).
// ---------------------------------------------------------------------------

type CanvasLayers = CanvasDocumentContractV2['layers'];

const layerExists = (layers: CanvasLayers, id: string): boolean => layers.some((layer) => layer.id === id);

/** Matches the auto-generated `Layer N` default name (N a positive integer). */
const AUTO_LAYER_NAME_PATTERN = /^Layer (\d+)$/;

/**
 * Picks the next auto-generated `Layer N` name that does not collide with any
 * existing name. Deriving `N` from the layer *count* (the previous approach)
 * collides after deletions — e.g. deleting `Layer 1` from `[Layer 1, Layer 2]`
 * leaves one layer, so the next add would re-mint `Layer 2`. This instead scans
 * the existing `Layer N` names and returns the lowest unused positive `N`, so
 * names stay unique regardless of add/delete history.
 */
export const nextLayerName = (existingNames: readonly string[]): string => {
  const used = new Set<number>();
  for (const name of existingNames) {
    const match = AUTO_LAYER_NAME_PATTERN.exec(name.trim());
    if (match) {
      const n = Number(match[1]);
      if (Number.isInteger(n) && n > 0) {
        used.add(n);
      }
    }
  }
  let n = 1;
  while (used.has(n)) {
    n += 1;
  }
  return `Layer ${n}`;
};

/**
 * Repairs a dangling `selectedLayerId` after a wholesale document swap
 * (`replaceCanvasDocument` / snapshot restore): if it names no layer present in
 * the incoming document, fall back to the top-most layer id (or null when the
 * document has no layers). Upholds the invariant that `selectedLayerId` is
 * always null or a present layer id.
 */
const repairSelectedLayerId = (document: CanvasDocumentContractV2): CanvasDocumentContractV2 =>
  document.selectedLayerId === null || layerExists(document.layers, document.selectedLayerId)
    ? document
    : { ...document, selectedLayerId: document.layers[0]?.id ?? null };

/** Returns a project whose canvas document is replaced, preserving identity when unchanged. */
const setCanvasDocument = (project: Project, document: CanvasDocumentContractV2): Project =>
  document === project.canvas.document ? project : { ...project, canvas: { ...project.canvas, document } };

const updateCanvasDocument = (
  project: Project,
  update: (document: CanvasDocumentContractV2) => CanvasDocumentContractV2
): Project => setCanvasDocument(project, update(project.canvas.document));

/** Returns a project whose canvas state is replaced, preserving identity when unchanged. */
const setCanvasState = (project: Project, canvas: CanvasStateContractV2): Project =>
  canvas === project.canvas ? project : { ...project, canvas };

/** Replaces exactly the layer with `id`, keeping every other layer's object identity. */
const mapCanvasLayer = (
  document: CanvasDocumentContractV2,
  id: string,
  update: (layer: CanvasLayerContract) => CanvasLayerContract
): CanvasDocumentContractV2 => {
  let changed = false;
  const layers = document.layers.map((layer) => {
    if (layer.id !== id) {
      return layer;
    }

    const next = update(layer);

    if (next !== layer) {
      changed = true;
    }

    return next;
  });

  return changed ? { ...document, layers } : document;
};

/**
 * Sets the `isEnabled` flag of many layers in one pass (the layers panel's
 * "hide/show all in group" action). Every listed layer is set to its target;
 * unlisted layers keep object identity. Returns the same document when nothing
 * actually changes so selectors don't churn.
 */
const setCanvasLayersEnabledInDocument = (
  document: CanvasDocumentContractV2,
  updates: readonly { id: string; isEnabled: boolean }[]
): CanvasDocumentContractV2 => {
  if (updates.length === 0) {
    return document;
  }
  const targets = new Map(updates.map((update) => [update.id, update.isEnabled]));
  let changed = false;
  const layers = document.layers.map((layer) => {
    const next = targets.get(layer.id);
    if (next === undefined || layer.isEnabled === next) {
      return layer;
    }
    changed = true;
    return { ...layer, isEnabled: next };
  });
  return changed ? { ...document, layers } : document;
};

/** After removing the selected layer, selects the nearest surviving layer below it, else above it, else null. */
const selectNearestRemainingLayerId = (
  previousLayers: CanvasLayers,
  remainingIds: Set<string>,
  removedSelectedId: string
): string | null => {
  const removedIndex = previousLayers.findIndex((layer) => layer.id === removedSelectedId);

  for (let index = removedIndex + 1; index < previousLayers.length; index += 1) {
    const id = previousLayers[index]?.id;

    if (id !== undefined && remainingIds.has(id)) {
      return id;
    }
  }

  for (let index = removedIndex - 1; index >= 0; index -= 1) {
    const id = previousLayers[index]?.id;

    if (id !== undefined && remainingIds.has(id)) {
      return id;
    }
  }

  return null;
};

const addCanvasLayerToDocument = (
  document: CanvasDocumentContractV2,
  layer: CanvasLayerContract,
  index = 0
): CanvasDocumentContractV2 => {
  const insertIndex = Math.min(Math.max(0, Math.round(index)), document.layers.length);

  return {
    ...document,
    layers: [...document.layers.slice(0, insertIndex), layer, ...document.layers.slice(insertIndex)],
    selectedLayerId: layer.id,
  };
};

const removeCanvasLayersFromDocument = (
  document: CanvasDocumentContractV2,
  ids: readonly string[]
): CanvasDocumentContractV2 => {
  const removed = new Set(ids);
  const layers = document.layers.filter((layer) => !removed.has(layer.id));

  if (layers.length === document.layers.length) {
    return document;
  }

  const remainingIds = new Set(layers.map((layer) => layer.id));
  const selectedLayerId =
    document.selectedLayerId !== null && removed.has(document.selectedLayerId)
      ? selectNearestRemainingLayerId(document.layers, remainingIds, document.selectedLayerId)
      : document.selectedLayerId;

  return { ...document, layers, selectedLayerId };
};

const duplicateCanvasLayerInDocument = (
  document: CanvasDocumentContractV2,
  sourceId: string,
  newId: string
): CanvasDocumentContractV2 => {
  const sourceIndex = document.layers.findIndex((layer) => layer.id === sourceId);

  if (sourceIndex === -1) {
    return document;
  }

  const source = document.layers[sourceIndex] as CanvasLayerContract;
  const duplicate = structuredClone(source);

  duplicate.id = newId;
  duplicate.name = `${source.name} copy`;

  return {
    ...document,
    layers: [...document.layers.slice(0, sourceIndex), duplicate, ...document.layers.slice(sourceIndex)],
    selectedLayerId: newId,
  };
};

const reorderCanvasLayersInDocument = (
  document: CanvasDocumentContractV2,
  orderedIds: readonly string[]
): CanvasDocumentContractV2 => {
  if (orderedIds.length !== document.layers.length || new Set(orderedIds).size !== orderedIds.length) {
    return document;
  }

  const byId = new Map(document.layers.map((layer) => [layer.id, layer]));
  const layers: CanvasLayers = [];

  for (const id of orderedIds) {
    const layer = byId.get(id);

    if (!layer) {
      return document;
    }

    layers.push(layer);
  }

  return { ...document, layers };
};

const applyCanvasLayerBasePatch = (layer: CanvasLayerContract, patch: CanvasLayerBasePatch): CanvasLayerContract => {
  const { transform, ...rest } = patch;

  return {
    ...layer,
    ...rest,
    transform: transform ? { ...layer.transform, ...transform } : layer.transform,
  };
};

const applyCanvasLayerConfigPatch = (
  layer: CanvasLayerContract,
  config: CanvasLayerConfigPatch
): CanvasLayerContract => {
  if (layer.type !== config.layerType) {
    return layer;
  }

  if (layer.type === 'raster' && config.layerType === 'raster') {
    return {
      ...layer,
      ...(config.adjustments !== undefined ? { adjustments: config.adjustments } : {}),
      ...(config.isTransparencyLocked !== undefined ? { isTransparencyLocked: config.isTransparencyLocked } : {}),
    };
  }

  if (layer.type === 'control' && config.layerType === 'control') {
    return {
      ...layer,
      ...(config.adapter ? { adapter: { ...layer.adapter, ...config.adapter } } : {}),
      ...(config.withTransparencyEffect !== undefined ? { withTransparencyEffect: config.withTransparencyEffect } : {}),
      ...(config.filter !== undefined ? { filter: config.filter } : {}),
    };
  }

  if (layer.type === 'regional_guidance' && config.layerType === 'regional_guidance') {
    return {
      ...layer,
      ...(config.mask ? { mask: { ...layer.mask, ...config.mask } } : {}),
      ...(config.positivePrompt !== undefined ? { positivePrompt: config.positivePrompt } : {}),
      ...(config.negativePrompt !== undefined ? { negativePrompt: config.negativePrompt } : {}),
      ...(config.autoNegative !== undefined ? { autoNegative: config.autoNegative } : {}),
      ...(config.referenceImages !== undefined ? { referenceImages: config.referenceImages } : {}),
    };
  }

  if (layer.type === 'inpaint_mask' && config.layerType === 'inpaint_mask') {
    return {
      ...layer,
      ...(config.mask ? { mask: { ...layer.mask, ...config.mask } } : {}),
      ...(config.noiseLevel !== undefined ? { noiseLevel: config.noiseLevel } : {}),
      ...(config.denoiseLimit !== undefined ? { denoiseLimit: config.denoiseLimit } : {}),
    };
  }

  return layer;
};

const updateCanvasLayerSourceInDocument = (
  document: CanvasDocumentContractV2,
  id: string,
  source: CanvasLayerSourceContract
): CanvasDocumentContractV2 =>
  mapCanvasLayer(document, id, (layer) =>
    layer.type === 'raster' || layer.type === 'control' ? { ...layer, source } : layer
  );

const convertCanvasLayerInDocument = (
  document: CanvasDocumentContractV2,
  id: string,
  targetType: CanvasLayerContract['type'],
  layer: CanvasLayerContract
): CanvasDocumentContractV2 => {
  if (layer.type !== targetType || !layerExists(document.layers, id)) {
    return document;
  }

  const converted = structuredClone(layer);

  converted.id = id;

  return { ...document, layers: document.layers.map((existing) => (existing.id === id ? converted : existing)) };
};

const mergeCanvasLayersDownInDocument = (
  document: CanvasDocumentContractV2,
  upperLayerId: string,
  source: Extract<CanvasLayerSourceContract, { type: 'paint' }>
): CanvasDocumentContractV2 => {
  const upperIndex = document.layers.findIndex((layer) => layer.id === upperLayerId);
  const below = upperIndex === -1 ? undefined : document.layers[upperIndex + 1];

  if (!below) {
    return document;
  }

  const mergedBelow: CanvasRasterLayerContractV2 = {
    blendMode: below.blendMode,
    id: below.id,
    isEnabled: below.isEnabled,
    isLocked: below.isLocked,
    name: below.name,
    opacity: below.opacity,
    source,
    transform: below.transform,
    type: 'raster',
  };
  const layers = document.layers
    .filter((_, index) => index !== upperIndex)
    .map((layer) => (layer.id === below.id ? mergedBelow : layer));
  const selectedLayerId = document.selectedLayerId === upperLayerId ? below.id : document.selectedLayerId;

  return { ...document, layers, selectedLayerId };
};

const clampCanvasBbox = (
  bbox: CanvasDocumentContractV2['bbox'],
  width: number,
  height: number
): CanvasDocumentContractV2['bbox'] => {
  const clampedWidth = Math.min(Math.max(1, Math.round(bbox.width)), width);
  const clampedHeight = Math.min(Math.max(1, Math.round(bbox.height)), height);

  return {
    height: clampedHeight,
    width: clampedWidth,
    x: Math.min(Math.max(0, Math.round(bbox.x)), width - clampedWidth),
    y: Math.min(Math.max(0, Math.round(bbox.y)), height - clampedHeight),
  };
};

const setCanvasBboxInDocument = (
  document: CanvasDocumentContractV2,
  bbox: CanvasDocumentContractV2['bbox']
): CanvasDocumentContractV2 => ({
  ...document,
  bbox: {
    height: Math.max(1, Math.round(bbox.height)),
    width: Math.max(1, Math.round(bbox.width)),
    x: Math.round(bbox.x),
    y: Math.round(bbox.y),
  },
});

const resizeCanvasDocumentInDocument = (
  document: CanvasDocumentContractV2,
  width: number,
  height: number,
  offsetX: number,
  offsetY: number
): CanvasDocumentContractV2 => {
  const nextWidth = Math.max(1, Math.round(width));
  const nextHeight = Math.max(1, Math.round(height));
  const layers =
    offsetX === 0 && offsetY === 0
      ? document.layers
      : document.layers.map((layer) => ({
          ...layer,
          transform: { ...layer.transform, x: layer.transform.x + offsetX, y: layer.transform.y + offsetY },
        }));

  return {
    ...document,
    bbox: clampCanvasBbox(
      { ...document.bbox, x: document.bbox.x + offsetX, y: document.bbox.y + offsetY },
      nextWidth,
      nextHeight
    ),
    height: nextHeight,
    layers,
    width: nextWidth,
  };
};

const saveCanvasSnapshotToState = (
  canvas: CanvasStateContractV2,
  snapshot: { id: string; name: string; createdAt: string }
): CanvasStateContractV2 => ({
  ...canvas,
  snapshots: [
    ...canvas.snapshots,
    { createdAt: snapshot.createdAt, document: structuredClone(canvas.document), id: snapshot.id, name: snapshot.name },
  ],
});

const restoreCanvasSnapshotInState = (canvas: CanvasStateContractV2, snapshotId: string): CanvasStateContractV2 => {
  const snapshot = canvas.snapshots.find((entry) => entry.id === snapshotId);

  // Wholesale swap: bump the revision so the document mirror clears engine pixel
  // history even when the restored document keeps the same dims/layer ids.
  return snapshot
    ? {
        ...canvas,
        document: repairSelectedLayerId(structuredClone(snapshot.document)),
        documentRevision: canvas.documentRevision + 1,
      }
    : canvas;
};

const deleteCanvasSnapshotInState = (canvas: CanvasStateContractV2, snapshotId: string): CanvasStateContractV2 => {
  const snapshots = canvas.snapshots.filter((entry) => entry.id !== snapshotId);

  return snapshots.length === canvas.snapshots.length ? canvas : { ...canvas, snapshots };
};

const getGalleryImages = (values: Record<string, unknown>): GeneratedImageContract[] =>
  Array.isArray(values.recentImages) ? (values.recentImages as GeneratedImageContract[]) : [];

const getGallerySelectedImageNames = (values: Record<string, unknown>): string[] => {
  if (Array.isArray(values.selectedImageNames)) {
    return (values.selectedImageNames as unknown[]).filter((name): name is string => typeof name === 'string');
  }

  return typeof values.selectedImageName === 'string' ? [values.selectedImageName] : [];
};

/**
 * Deep-clones an already-v2 canvas state and normalizes staging candidate placements. Not a
 * migration boundary: callers with genuinely unknown/legacy input must run
 * `migrateCanvasStateToV2` first (see `ensureProjectWidgetContracts`).
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
const createUndoSnapshot = (project: Project): ProjectUndoSnapshot => ({
  invocation: { ...project.invocation },
  layout: { ...project.layout, panels: { ...project.layout.panels } },
  projectGraph: cloneProjectGraph(project.projectGraph),
  widgetGraphs: cloneWidgetGraphs(project.widgetGraphs),
  widgetInstances: cloneWidgetInstances(project.widgetInstances),
  widgetRegions: cloneWidgetRegions(project.widgetRegions),
});

const restoreUndoSnapshot = (project: Project, snapshot: ProjectUndoSnapshot): Project => ({
  ...project,
  invocation: { ...snapshot.invocation },
  layout: { ...snapshot.layout, panels: { ...snapshot.layout.panels } },
  projectGraph: cloneProjectGraph(normalizeProjectGraph(snapshot.projectGraph)),
  widgetGraphs: cloneWidgetGraphs(snapshot.widgetGraphs),
  widgetInstances: cloneWidgetInstances(snapshot.widgetInstances),
  widgetRegions: cloneWidgetRegions(snapshot.widgetRegions),
});

const createGraphHistorySnapshot = (label: string, graph: GraphContract): GraphHistorySnapshot => ({
  createdAt: now(),
  graph: cloneGraph(graph),
  id: createId('graph-history'),
  label,
});

/** A restorable history entry carrying the editable workflow document. */
const createDocumentHistorySnapshot = (label: string, document: ProjectGraphState): GraphHistorySnapshot => ({
  createdAt: now(),
  document: cloneProjectGraph(document),
  id: createId('graph-history'),
  label,
});

const pushUndo = (project: Project, label: string): Project => ({
  ...project,
  undoRedo: {
    future: [],
    past: [
      ...project.undoRedo.past,
      {
        createdAt: now(),
        id: createId('undo'),
        label,
        project: createUndoSnapshot(project),
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
  layers: { id: 'layers', label: 'Layers', values: {}, version: 1 },
  notifications: { id: 'notifications', label: 'Notifications', values: {}, version: 1 },
  preview: { id: 'preview', label: 'Preview', values: {}, version: 1 },
  project: { id: 'project', label: 'Project', values: {}, version: 1 },
  queue: { id: 'queue', label: 'Queue', values: {}, version: 1 },
  'server-status': { id: 'server-status', label: 'Server Status', values: {}, version: 1 },
  users: { id: 'users', label: 'Users', values: {}, version: 1 },
  'version-status': { id: 'version-status', label: 'Version', values: {}, version: 1 },
  workflow: { graphId: 'workflow-graph', id: 'workflow', label: 'Workflow', values: {}, version: 1 },
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

const LEGACY_RIGHT_REGION_WIDGET_IDS: WidgetId[] = ['queue', 'gallery', 'layers'];

const isLegacyDefaultRightRegion = (region: WidgetRegionState): boolean =>
  region.instanceIds.length === LEGACY_RIGHT_REGION_WIDGET_IDS.length &&
  region.instanceIds.every((widgetId, index) => widgetId === LEGACY_RIGHT_REGION_WIDGET_IDS[index]);

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

const ensureProjectWidgetContracts = (project: Project): Project => {
  const defaultWidgetRegions = createWidgetRegions();
  const legacyWidgetRegions = project.widgetRegions as
    | Partial<Record<WidgetRegion | 'left-panel' | 'right-panel' | 'status-bar', WidgetRegionState>>
    | undefined;
  const widgetInstances = project.widgetInstances ?? createWidgetInstances();

  return {
    ...project,
    // `project` may come straight from persisted storage (an unsafe cast boundary), so its
    // canvas can still be v1-shaped, malformed, or missing — migrate before cloning.
    canvas: cloneCanvas(migrateCanvasStateToV2(project.canvas)),
    projectGraph: normalizeProjectGraph(project.projectGraph),
    promptHistory: normalizePromptHistory((project as Partial<Project>).promptHistory),
    settings: normalizeProjectSettings(project.settings),
    widgetRegions: {
      left: legacyWidgetRegions?.left ?? legacyWidgetRegions?.['left-panel'] ?? defaultWidgetRegions.left,
      right: ensureRightRegion(legacyWidgetRegions?.right ?? legacyWidgetRegions?.['right-panel']),
      bottom: legacyWidgetRegions?.bottom ?? legacyWidgetRegions?.['status-bar'] ?? defaultWidgetRegions.bottom,
      center: ensureCenterRegion(legacyWidgetRegions?.center, project.layout.centerViewId),
    },
    widgetInstances: cloneWidgetInstances(widgetInstances),
  };
};

const clampPanelSize = (region: WidgetRegion, sizePx: number): number => {
  if (region === 'bottom') {
    return Math.min(MAX_STATUS_PANEL_SIZE_PX, Math.max(MIN_STATUS_PANEL_SIZE_PX, sizePx));
  }

  return Math.min(MAX_PANEL_SIZE_PX, Math.max(MIN_PANEL_SIZE_PX, sizePx));
};

const createCanvasState = (): CanvasStateContractV2 => createNewCanvasStateV2();

const createProject = (index: number, id = `project-${index}`): Project => ({
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
  invocation: { ...defaultInvocationRoute },
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
});

const getNextProjectIndex = (projects: Project[]): number => {
  const usedIndices = projects.map((project) => Number(project.name.match(/#(\d+)$/)?.[1] ?? 0));

  return Math.max(0, ...usedIndices) + 1;
};

/**
 * A fresh, never-saved project. Ids carry entropy rather than an index so a
 * draft can never collide with a project that already exists on the server
 * (which an autosave would then silently overwrite).
 */
export const createDraftProject = (projects: Project[]): Project =>
  createProject(getNextProjectIndex(projects), createId('project'));

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
  layout: { ...snapshot.layout, panels: { ...snapshot.layout.panels } },
  widgetInstances: Object.fromEntries(
    Object.entries(snapshot.widgetInstances).map(([instanceId, instance]) => [instanceId, { ...instance }])
  ),
  widgetRegions: cloneLayoutPresetWidgetRegions(snapshot.widgetRegions),
});

const isWidgetRegionState = (value: unknown): value is WidgetRegionState => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const record = value as Partial<WidgetRegionState>;

  return (
    typeof record.activeInstanceId === 'string' &&
    Array.isArray(record.instanceIds) &&
    record.instanceIds.every((instanceId) => typeof instanceId === 'string') &&
    typeof record.isCollapsed === 'boolean' &&
    typeof record.sizePx === 'number'
  );
};

const isLayoutPresetSnapshot = (value: unknown): value is LayoutPresetSnapshot => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const snapshot = value as Partial<LayoutPresetSnapshot>;
  const layout = snapshot.layout as Partial<ProjectLayoutState> | undefined;

  return (
    !!layout &&
    typeof layout.presetId === 'string' &&
    typeof layout.centerViewId === 'string' &&
    !!layout.panels &&
    typeof layout.panels.isBottomOpen === 'boolean' &&
    typeof layout.panels.isLeftOpen === 'boolean' &&
    typeof layout.panels.isRightOpen === 'boolean' &&
    !!snapshot.widgetInstances &&
    typeof snapshot.widgetInstances === 'object' &&
    !!snapshot.widgetRegions &&
    isWidgetRegionState(snapshot.widgetRegions.left) &&
    isWidgetRegionState(snapshot.widgetRegions.right) &&
    isWidgetRegionState(snapshot.widgetRegions.bottom) &&
    isWidgetRegionState(snapshot.widgetRegions.center)
  );
};

const normalizeCustomLayoutPresets = (presets: unknown): LayoutPreset[] => {
  if (!Array.isArray(presets)) {
    return [];
  }

  return presets.flatMap((preset): LayoutPreset[] => {
    if (!preset || typeof preset !== 'object') {
      return [];
    }

    const record = preset as Partial<LayoutPreset>;

    if (typeof record.id !== 'string' || typeof record.label !== 'string' || !isLayoutPresetSnapshot(record.snapshot)) {
      return [];
    }

    return [
      {
        id: record.id,
        label: record.label,
        snapshot: cloneLayoutPresetSnapshot(record.snapshot),
      },
    ];
  });
};

const normalizeAccount = (account: Partial<WorkbenchState['account']> | undefined): WorkbenchState['account'] => ({
  activeLayoutPresetId: account?.activeLayoutPresetId ?? defaultLayoutPreset.id,
  customLayoutPresets: normalizeCustomLayoutPresets(account?.customLayoutPresets),
});

const normalizeWorkbenchState = (state: WorkbenchState): WorkbenchState => ({
  ...state,
  backendConnection: { status: 'connecting' },
  // Built explicitly: legacy snapshots carried preferences inside the account
  // (they live in the settings store now) and must not resurface here.
  account: normalizeAccount(state.account),
  notifications: [],
  projects: state.projects.map(ensureProjectWidgetContracts),
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
  state.account.customLayoutPresets?.find((preset) => preset.id === presetId) ?? getLayoutPreset(presetId);

const applyLayoutPresetToProject = (project: Project, preset: LayoutPreset): Project => {
  const snapshot = preset.snapshot;
  const widgetInstances = { ...project.widgetInstances };

  for (const instance of Object.values(snapshot.widgetInstances)) {
    widgetInstances[instance.id] = widgetInstances[instance.id]
      ? { ...widgetInstances[instance.id], title: instance.title }
      : createWidgetInstance(instance.typeId, instance.id);
  }

  return {
    ...project,
    layout: {
      ...snapshot.layout,
      panels: { ...snapshot.layout.panels },
      presetId: preset.id,
    },
    widgetInstances,
    widgetRegions: cloneLayoutPresetWidgetRegions(snapshot.widgetRegions),
  };
};

const updateActiveProjectLayoutPreset = (state: WorkbenchState, preset: LayoutPreset): WorkbenchState =>
  updateActiveProject(state, (project) => {
    const nextProject = pushUndo(project, 'Update layout');

    return {
      ...applyLayoutPresetToProject(nextProject, preset),
      events: [
        {
          createdAt: now(),
          id: createId('event'),
          summary: 'Updated active layout',
          type: 'layout-updated',
        },
        ...nextProject.events,
      ],
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

const refreshProjectBackendData = (project: Project): Project => ({
  ...updateProjectWidgetValues(project, 'gallery', (values) => ({
    ...values,
    galleryImagesRefreshToken: createId('gallery-images-refresh'),
    galleryRefreshToken: createId('gallery-refresh'),
  })),
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
): QueueItemStatus => {
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
  const existingImages = getGalleryImages(galleryValues).filter(
    (image) => !images.some((incomingImage) => incomingImage.imageName === image.imageName)
  );

  return updateProjectWidgetValues(project, 'gallery', () => ({
    ...galleryValues,
    recentImages: [...images, ...existingImages],
    selectedImage: images[0] ?? galleryValues.selectedImage,
    selectedImageName: images[0]?.imageName ?? galleryValues.selectedImageName,
    selectedImageNames: images[0] ? [images[0].imageName] : getGallerySelectedImageNames(galleryValues),
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

  return destination === 'gallery' ? updateGalleryWithResultImages(nextProject, images) : nextProject;
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

  const incomingImages = images.map((image) => normalizeStagingCandidate(image, nextProject.canvas.document));
  const incomingImageKeys = new Set(incomingImages.map((image) => `${image.sourceQueueItemId}:${image.imageName}`));
  const existingImages = nextProject.canvas.stagingArea.pendingImages.filter(
    (image) => !incomingImageKeys.has(`${image.sourceQueueItemId}:${image.imageName}`)
  );
  const pendingImages = [...existingImages, ...incomingImages];

  return {
    ...nextProject,
    canvas: {
      ...nextProject.canvas,
      stagingArea: {
        ...nextProject.canvas.stagingArea,
        areThumbnailsVisible: true,
        isVisible: pendingImages.length > 0,
        pendingImageIds: pendingImages.map((image) => image.imageName),
        pendingImages,
        selectedImageIndex: resolveStagingSelectionIndex(
          nextProject.canvas.stagingArea.autoSwitchMode,
          nextProject.canvas.stagingArea.selectedImageIndex,
          existingImages.length,
          incomingImages.length,
          pendingImages.length
        ),
        sourceQueueItemId: queueItemId,
      },
    },
  };
};

/**
 * Enqueues an already-compiled graph snapshot. Shared by the route-validated
 * `submitInvocationSnapshot` and the canvas engine's `submitCanvasInvocationSnapshot`,
 * whose graph is compiled asynchronously outside the reducer.
 */
const enqueueCompiledSnapshot = (
  project: Project,
  route: InvocationRoute,
  compiled: { generate?: QueueGenerateSnapshot; graph: GraphContract; widgetStates: WidgetStateMap },
  backendSupportsCancellation: boolean
): Project => {
  const submittedAt = now();
  const queueItemId = createId('queue-item');
  const { generate, graph } = compiled;
  const widgetStates = applyQueueGenerateSnapshotToWidgetStates(compiled.widgetStates, generate);
  const graphHistorySnapshot = createGraphHistorySnapshot(`Queue snapshot ${queueItemId}`, graph);
  const generateSettings =
    route.sourceId === 'generate' ? normalizeGenerateSettings(widgetStates.generate.values) : null;
  const queueItem: QueueItem = {
    cancellable: backendSupportsCancellation,
    id: queueItemId,
    snapshot: {
      canvas: cloneCanvas(project.canvas),
      destination: route.destination,
      ...(generate ? { generate: cloneQueueGenerateSnapshot(generate) } : {}),
      graph,
      sourceId: route.sourceId,
      submittedAt,
      widgetInstances: cloneWidgetInstances(project.widgetInstances),
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
    graphHistory: [graphHistorySnapshot, ...project.graphHistory].slice(0, HISTORY_LIMIT),
    promptHistory: generateSettings
      ? addPromptHistoryItem(project.promptHistory, getPromptHistoryItemFromGenerateSettings(generateSettings))
      : project.promptHistory,
    invocation: {
      ...project.invocation,
      destination: route.destination,
      lastSubmittedRunId: queueItemId,
      sourceId: route.sourceId,
    },
    queue: { items: [queueItem, ...project.queue.items] },
    widgetGraphs:
      route.sourceId === 'generate' ? { ...project.widgetGraphs, generate: cloneGraph(graph) } : project.widgetGraphs,
  };
};

const submitInvocationSnapshot = (
  project: Project,
  backendSupportsCancellation: boolean,
  route = resolveInvocationRoute(project),
  models?: readonly ModelConfig[]
): Project => {
  if (!isInvocationRouteValid(route)) {
    return project;
  }

  const compiledSnapshot = compileInvocationSnapshot(project, route, models);

  if (!compiledSnapshot) {
    return project;
  }

  return enqueueCompiledSnapshot(project, route, compiledSnapshot, backendSupportsCancellation);
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

export const workbenchReducer = (state: WorkbenchState, action: WorkbenchAction): WorkbenchState => {
  switch (action.type) {
    case 'createProject': {
      const project = createDraftProject(state.projects);

      return { ...state, activeProjectId: project.id, projects: [...state.projects, project] };
    }
    case 'openProject': {
      // Hydrated from the library (Open dialog or a deep link). Opening an
      // already-open project just focuses its tab.
      if (state.projects.some((project) => project.id === action.project.id)) {
        return { ...state, activeProjectId: action.project.id };
      }

      const project = ensureProjectWidgetContracts(action.project);

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
      const nextState = updateActiveProjectLayoutPreset(state, preset);

      return {
        ...nextState,
        account: { ...state.account, activeLayoutPresetId: preset.id },
      };
    }
    case 'addLayoutPreset': {
      const activeProject = state.projects.find((project) => project.id === state.activeProjectId);

      if (!activeProject) {
        return state;
      }

      const preset: LayoutPreset = {
        id: action.presetId,
        label: action.label.trim() || 'Custom layout',
        snapshot: createLayoutPresetSnapshot(ensureProjectWidgetContracts(activeProject)),
      };
      const customLayoutPresets = [
        ...(state.account.customLayoutPresets ?? []).filter((candidate) => candidate.id !== action.presetId),
        preset,
      ];

      return {
        ...state,
        account: { ...state.account, activeLayoutPresetId: preset.id, customLayoutPresets },
      };
    }
    case 'renameLayoutPreset': {
      const label = action.label.trim();

      if (!label) {
        return state;
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
      const customLayoutPresets = (state.account.customLayoutPresets ?? []).filter(
        (preset) => preset.id !== action.presetId
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
        },
      };
    }
    case 'resetActiveLayout': {
      const preset = getAvailableLayoutPreset(
        state,
        state.projects.find((project) => project.id === state.activeProjectId)?.layout.presetId ??
          state.account.activeLayoutPresetId
      );

      return updateActiveProjectLayoutPreset(state, preset);
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

        return {
          ...project,
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
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) =>
        updateProjectWidgetValues(project, 'generate', () => cloneGenerateWidgetValues(action.values))
      );
    }
    case 'patchGenerateSettings': {
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) =>
        updateProjectWidgetValues(project, 'generate', (values) => patchRecord(values, action.values))
      );
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
      return updateProjectById(state, action.projectId ?? state.activeProjectId, (project) =>
        updateProjectWidgetValues(project, action.widgetId, (values) => patchRecord(values, action.values))
      );
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

        const undoLabel = getProjectGraphUndoLabel(action.action);
        const nextProject = undoLabel ? pushUndo(project, undoLabel) : project;
        // Meaningful workflow edits are high-confidence source events: they
        // steer the global Invoke route to the project graph unless locked.
        const shouldAutoSetSource =
          !project.invocation.sourceLocked &&
          project.invocation.sourceId !== 'workflow' &&
          isHighConfidenceGraphEdit(action.action) &&
          isInvocationSourceAvailable('workflow');

        return {
          ...nextProject,
          invocation: shouldAutoSetSource
            ? { ...nextProject.invocation, sourceId: 'workflow' }
            : nextProject.invocation,
          projectGraph,
        };
      });
    }
    case 'replaceProjectGraph': {
      const nextState = updateActiveProject(state, (project) => {
        const nextProject = pushUndo(project, 'Replace project graph');

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
          graphHistory: [
            createDocumentHistorySnapshot(`Before: ${action.label}`, project.projectGraph),
            ...nextProject.graphHistory,
          ].slice(0, HISTORY_LIMIT),
          projectGraph: cloneProjectGraph(action.document),
        };
      });
      const activeProject = nextState.projects.find((project) => project.id === nextState.activeProjectId);

      return addNotification(
        nextState,
        createNotification({
          kind: 'info',
          message: `The previous project graph was saved to graph history.`,
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
        graphHistory: [
          createDocumentHistorySnapshot(
            `Manual save: ${project.projectGraph.name || 'Untitled Workflow'}`,
            project.projectGraph
          ),
          ...project.graphHistory,
        ].slice(0, HISTORY_LIMIT),
      }));
    }
    case 'restoreProjectGraphSnapshot': {
      return updateActiveProject(state, (project) => {
        const snapshot = project.graphHistory.find((entry) => entry.id === action.snapshotId);

        if (!snapshot?.document) {
          return project;
        }

        const nextProject = pushUndo(project, 'Restore graph history snapshot');

        return {
          ...nextProject,
          graphHistory: [
            createDocumentHistorySnapshot('Before restore', project.projectGraph),
            ...nextProject.graphHistory,
          ].slice(0, HISTORY_LIMIT),
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
      const beforeProject = state.projects.find((project) => project.id === state.activeProjectId);
      const nextState = updateActiveProject(state, (project) =>
        submitInvocationSnapshot(project, action.backendSupportsCancellation, undefined, action.models)
      );
      const afterProject = nextState.projects.find((project) => project.id === nextState.activeProjectId);

      if (!beforeProject || !afterProject || beforeProject.queue.items.length === afterProject.queue.items.length) {
        return nextState;
      }

      const queueItem = afterProject.queue.items[0];

      return addNotification(
        nextState,
        createNotification({
          kind: 'info',
          message: `${afterProject.name}: ${queueItem.snapshot.sourceId} to ${queueItem.snapshot.destination}`,
          projectId: afterProject.id,
          title: 'Invocation queued',
        })
      );
    }
    case 'submitResolvedInvocationSnapshot': {
      return updateActiveProject(state, (project) =>
        submitInvocationSnapshot(
          project,
          action.backendSupportsCancellation,
          resolveInvocationRoute(project, 'global', action.route, action.models),
          action.models
        )
      );
    }
    case 'markQueueItemBackendSubmitted': {
      return updateProjectById(state, action.projectId, (project) =>
        updateQueueItem(project, action.queueItemId, (item) => {
          const status = item.status === 'cancelled' ? 'cancelled' : 'running';
          const hasSameBackendItemIds =
            item.backendItemIds?.length === action.backendItemIds.length &&
            item.backendItemIds.every((id, index) => id === action.backendItemIds[index]);

          return item.backendBatchId === action.backendBatchId && hasSameBackendItemIds && item.status === status
            ? item
            : { ...item, backendBatchId: action.backendBatchId, backendItemIds: action.backendItemIds, status };
        })
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
        updateQueueItem(project, action.queueItemId, (item) => ({
          ...item,
          error: action.error,
          status: action.status,
        }))
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

      return updateProjectById(state, action.projectId, (project) =>
        updateQueueItem(project, action.queueItemId, (item) => {
          const cancelledBackendItemIds = mergeBackendItemId(item.cancelledBackendItemIds, action.backendItemId);

          return {
            ...item,
            cancelledBackendItemIds,
            status: getQueueItemStatusAfterBackendCancellation(item, cancelledBackendItemIds),
          };
        })
      );
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
    case 'setStagedImageIndex': {
      return updateActiveProject(state, (project) => {
        const selectedImageIndex = clampStagedImageIndex(
          action.imageIndex,
          project.canvas.stagingArea.pendingImages.length
        );

        return {
          ...project,
          canvas: {
            ...project.canvas,
            stagingArea: { ...project.canvas.stagingArea, selectedImageIndex },
          },
        };
      });
    }
    case 'cycleStagedImage': {
      return updateActiveProject(state, (project) => {
        const { pendingImages, selectedImageIndex } = project.canvas.stagingArea;

        return {
          ...project,
          canvas: {
            ...project.canvas,
            stagingArea: {
              ...project.canvas.stagingArea,
              selectedImageIndex: cycleStagedImageIndex(selectedImageIndex, pendingImages.length, action.direction),
            },
          },
        };
      });
    }
    case 'discardSelectedStagedImage': {
      return updateActiveProject(state, (project) => {
        const { pendingImages, selectedImageIndex } = project.canvas.stagingArea;

        if (pendingImages.length === 0) {
          return project;
        }

        const nextPendingImages = pendingImages.filter((_image, index) => index !== selectedImageIndex);

        return {
          ...project,
          canvas: {
            ...project.canvas,
            stagingArea: {
              ...project.canvas.stagingArea,
              isVisible: nextPendingImages.length > 0 ? project.canvas.stagingArea.isVisible : false,
              pendingImageIds: nextPendingImages.map((image) => image.imageName),
              pendingImages: nextPendingImages,
              selectedImageIndex: clampStagedImageIndex(selectedImageIndex, nextPendingImages.length),
              sourceQueueItemId:
                nextPendingImages.length > 0 ? project.canvas.stagingArea.sourceQueueItemId : undefined,
            },
          },
        };
      });
    }
    case 'discardAllStagedImages': {
      return updateActiveProject(state, (project) => ({
        ...project,
        canvas: { ...project.canvas, stagingArea: clearStagingArea(project.canvas.stagingArea) },
      }));
    }
    case 'selectGalleryImage': {
      return updateGalleryValues(
        state,
        (values) => ({
          ...values,
          selectedImage: action.image,
          selectedImageName: action.image.imageName,
          selectedImageNames: [action.image.imageName],
        }),
        action.projectId
      );
    }
    case 'toggleGalleryImageInSelection': {
      return updateGalleryValues(
        state,
        (values) => {
          const imageName = action.image.imageName;
          const selectedImageNames = getGallerySelectedImageNames(values);

          if (!selectedImageNames.includes(imageName)) {
            return {
              ...values,
              selectedImage: action.image,
              selectedImageName: imageName,
              selectedImageNames: [...selectedImageNames, imageName],
            };
          }

          const remainingImageNames = selectedImageNames.filter((name) => name !== imageName);
          const wasPrimary = values.selectedImageName === imageName;

          return {
            ...values,
            selectedImage: wasPrimary ? null : values.selectedImage,
            selectedImageName: wasPrimary
              ? (remainingImageNames[remainingImageNames.length - 1] ?? null)
              : values.selectedImageName,
            selectedImageNames: remainingImageNames,
          };
        },
        action.projectId
      );
    }
    case 'setGalleryMultiSelection': {
      return updateGalleryValues(
        state,
        (values) => ({
          ...values,
          selectedImage: action.primaryImage,
          selectedImageName: action.primaryImage.imageName,
          selectedImageNames: action.imageNames,
        }),
        action.projectId
      );
    }
    case 'setGalleryCompareImage': {
      return updateGalleryValues(state, (values) => ({ ...values, compareImage: action.image }), action.projectId);
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
      const resetsQuery =
        action.settings.imageOrderDir !== undefined ||
        action.settings.starredFirst !== undefined ||
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
    case 'touchGalleryRefresh': {
      return updateGalleryValues(
        state,
        (values) => ({
          ...values,
          galleryImagesRefreshToken: createId('gallery-images-refresh'),
          galleryRefreshToken: createId('gallery-refresh'),
        }),
        action.projectId
      );
    }
    case 'touchGalleryImagesRefresh': {
      return updateGalleryValues(
        state,
        (values) => ({
          ...values,
          galleryImagesRefreshToken: createId('gallery-images-refresh'),
        }),
        action.projectId
      );
    }
    case 'removeGalleryImages': {
      const removedImageNames = new Set(action.imageNames);

      return updateGalleryValues(
        state,
        (values) => {
          const selectedImage = values.selectedImage as GeneratedImageContract | null | undefined;
          const compareImage = values.compareImage as GeneratedImageContract | null | undefined;
          const selectedImageName = typeof values.selectedImageName === 'string' ? values.selectedImageName : null;

          return {
            ...values,
            compareImage: compareImage && removedImageNames.has(compareImage.imageName) ? null : compareImage,
            recentImages: getGalleryImages(values).filter((image) => !removedImageNames.has(image.imageName)),
            selectedImage: selectedImage && removedImageNames.has(selectedImage.imageName) ? null : selectedImage,
            selectedImageName: selectedImageName && removedImageNames.has(selectedImageName) ? null : selectedImageName,
            selectedImageNames: getGallerySelectedImageNames(values).filter((name) => !removedImageNames.has(name)),
          };
        },
        action.projectId
      );
    }
    case 'setGalleryProjectBoardId': {
      return updateGalleryValues(state, (values) => ({ ...values, projectBoardId: action.boardId }), action.projectId);
    }
    case 'toggleCanvasStagingVisibility': {
      return updateActiveProject(state, (project) => {
        if (project.canvas.stagingArea.pendingImages.length === 0) {
          return project;
        }

        return {
          ...project,
          canvas: {
            ...project.canvas,
            stagingArea: { ...project.canvas.stagingArea, isVisible: !project.canvas.stagingArea.isVisible },
          },
        };
      });
    }
    case 'toggleCanvasStagingThumbnailsVisibility': {
      return updateActiveProject(state, (project) => {
        if (project.canvas.stagingArea.pendingImages.length === 0) {
          return project;
        }

        return {
          ...project,
          canvas: {
            ...project.canvas,
            stagingArea: {
              ...project.canvas.stagingArea,
              areThumbnailsVisible: !project.canvas.stagingArea.areThumbnailsVisible,
            },
          },
        };
      });
    }
    case 'acceptStagedImage': {
      const activeProject = state.projects.find((project) => project.id === state.activeProjectId);
      const stagedImage =
        activeProject?.canvas.stagingArea.pendingImages[activeProject.canvas.stagingArea.selectedImageIndex];
      const nextState = updateActiveProject(state, (project) => {
        const stagedImage = project.canvas.stagingArea.pendingImages[project.canvas.stagingArea.selectedImageIndex];

        if (!stagedImage) {
          return project;
        }

        const acceptedAt = now();
        // Canvas mutations are engine-owned and no longer participate in project
        // undo, so accept does not `pushUndo`. The layer lands at the bbox origin
        // (transform x/y = bbox x/y, unscaled), on top, and becomes selected.
        const { bbox } = project.canvas.document;
        const layer: CanvasRasterLayerContractV2 = {
          blendMode: 'normal',
          id: createId('layer'),
          isEnabled: true,
          isLocked: false,
          name: `Layer ${project.canvas.document.layers.length + 1}`,
          opacity: 1,
          source: {
            image: { height: stagedImage.height, imageName: stagedImage.imageName, width: stagedImage.width },
            type: 'image',
          },
          transform: { rotation: 0, scaleX: 1, scaleY: 1, x: bbox.x, y: bbox.y },
          type: 'raster',
        };

        return {
          ...project,
          canvas: {
            ...project.canvas,
            document: {
              ...project.canvas.document,
              layers: [layer, ...project.canvas.document.layers],
              selectedLayerId: layer.id,
            },
            stagingArea: clearStagingArea(project.canvas.stagingArea),
          },
          events: [
            {
              createdAt: acceptedAt,
              id: createId('event'),
              summary: `Accepted ${stagedImage.imageName} into a new raster layer`,
              type: 'canvas-layer-accepted',
            },
            ...project.events,
          ],
        };
      });

      if (!stagedImage || !activeProject) {
        return nextState;
      }

      return addNotification(
        nextState,
        createNotification({
          kind: 'success',
          message: `${stagedImage.imageName} added to ${activeProject.name}.`,
          projectId: activeProject.id,
          title: 'Canvas layer accepted',
        })
      );
    }
    case 'clearCanvasStaging': {
      return updateActiveProject(state, (project) => ({
        ...project,
        canvas: { ...project.canvas, stagingArea: clearStagingArea(project.canvas.stagingArea) },
      }));
    }
    case 'addCanvasLayer': {
      return updateActiveProject(state, (project) =>
        updateCanvasDocument(project, (document) => addCanvasLayerToDocument(document, action.layer, action.index))
      );
    }
    case 'removeCanvasLayers': {
      return updateActiveProject(state, (project) =>
        updateCanvasDocument(project, (document) => removeCanvasLayersFromDocument(document, action.ids))
      );
    }
    case 'duplicateCanvasLayer': {
      return updateActiveProject(state, (project) =>
        updateCanvasDocument(project, (document) =>
          duplicateCanvasLayerInDocument(document, action.sourceId, action.newId)
        )
      );
    }
    case 'reorderCanvasLayers': {
      return updateActiveProject(state, (project) =>
        updateCanvasDocument(project, (document) => reorderCanvasLayersInDocument(document, action.orderedIds))
      );
    }
    case 'updateCanvasLayer': {
      return updateActiveProject(state, (project) =>
        updateCanvasDocument(project, (document) =>
          mapCanvasLayer(document, action.id, (layer) => applyCanvasLayerBasePatch(layer, action.patch))
        )
      );
    }
    case 'setCanvasLayersEnabled': {
      return updateActiveProject(state, (project) =>
        updateCanvasDocument(project, (document) => setCanvasLayersEnabledInDocument(document, action.updates))
      );
    }
    case 'updateCanvasLayerSource': {
      return updateActiveProject(state, (project) =>
        updateCanvasDocument(project, (document) =>
          updateCanvasLayerSourceInDocument(document, action.id, action.source)
        )
      );
    }
    case 'updateCanvasLayerConfig': {
      return updateActiveProject(state, (project) =>
        updateCanvasDocument(project, (document) =>
          mapCanvasLayer(document, action.id, (layer) => applyCanvasLayerConfigPatch(layer, action.config))
        )
      );
    }
    case 'convertCanvasLayer': {
      return updateActiveProject(state, (project) =>
        updateCanvasDocument(project, (document) =>
          convertCanvasLayerInDocument(document, action.id, action.targetType, action.layer)
        )
      );
    }
    case 'mergeCanvasLayersDown': {
      return updateActiveProject(state, (project) =>
        updateCanvasDocument(project, (document) =>
          mergeCanvasLayersDownInDocument(document, action.upperLayerId, action.source)
        )
      );
    }
    case 'setCanvasBbox': {
      return updateActiveProject(state, (project) =>
        updateCanvasDocument(project, (document) => setCanvasBboxInDocument(document, action.bbox))
      );
    }
    case 'setCanvasSelectedLayer': {
      return updateActiveProject(state, (project) =>
        updateCanvasDocument(project, (document) => {
          if (action.id !== null && !layerExists(document.layers, action.id)) {
            return document;
          }

          return document.selectedLayerId === action.id ? document : { ...document, selectedLayerId: action.id };
        })
      );
    }
    case 'resizeCanvasDocument': {
      return updateActiveProject(state, (project) =>
        updateCanvasDocument(project, (document) =>
          resizeCanvasDocumentInDocument(
            document,
            action.width,
            action.height,
            action.offsetX ?? 0,
            action.offsetY ?? 0
          )
        )
      );
    }
    case 'replaceCanvasDocument': {
      return updateActiveProject(state, (project) =>
        setCanvasState(project, {
          ...project.canvas,
          document: repairSelectedLayerId(structuredClone(action.document)),
          // Wholesale swap: bump the revision so the document mirror clears engine
          // pixel history even when the new document keeps the same dims/layer ids.
          documentRevision: project.canvas.documentRevision + 1,
          // Staged candidates (and their pending ids) belong to the outgoing
          // document — a swap (new canvas session) must not leave them hovering
          // over the fresh document, nor let a late-completing generation append
          // stale candidates. Same rule as snapshot restore above.
          stagingArea: clearStagingArea(project.canvas.stagingArea),
        })
      );
    }
    case 'saveCanvasSnapshot': {
      return updateActiveProject(state, (project) =>
        setCanvasState(
          project,
          saveCanvasSnapshotToState(project.canvas, {
            createdAt: action.createdAt,
            id: action.id,
            name: action.name,
          })
        )
      );
    }
    case 'restoreCanvasSnapshot': {
      return updateActiveProject(state, (project) =>
        setCanvasState(project, restoreCanvasSnapshotInState(project.canvas, action.snapshotId))
      );
    }
    case 'deleteCanvasSnapshot': {
      return updateActiveProject(state, (project) =>
        setCanvasState(project, deleteCanvasSnapshotInState(project.canvas, action.snapshotId))
      );
    }
    case 'setCanvasStagingAutoSwitch': {
      return updateActiveProject(state, (project) =>
        project.canvas.stagingArea.autoSwitchMode === action.mode
          ? project
          : {
              ...project,
              canvas: {
                ...project.canvas,
                stagingArea: { ...project.canvas.stagingArea, autoSwitchMode: action.mode },
              },
            }
      );
    }
    case 'submitCanvasInvocationSnapshot': {
      return updateProjectById(state, action.projectId, (project) =>
        enqueueCompiledSnapshot(
          project,
          { ...project.invocation, destination: action.destination, sourceId: 'canvas' },
          {
            generate: action.generate,
            graph: action.graph,
            widgetStates: getWidgetStatesSnapshot(project.widgetInstances),
          },
          action.backendSupportsCancellation
        )
      );
    }
    case 'cancelQueueItem': {
      const targetProjectId = action.projectId ?? state.activeProjectId;
      const targetProject = state.projects.find((project) => project.id === targetProjectId);
      const queueItem = targetProject?.queue.items.find((item) => item.id === action.queueItemId);
      const canCancelQueueItem = queueItem ? isCancellableQueueItem(queueItem) : false;
      const nextState = updateProjectById(state, targetProjectId, (project) => ({
        ...project,
        queue: {
          items: project.queue.items.map((item) => {
            if (item.id !== action.queueItemId || !isCancellableQueueItem(item)) {
              return item;
            }

            return { ...item, status: 'cancelled' };
          }),
        },
      }));

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
        projects: state.projects.map((project) => ({
          ...project,
          queue: {
            items: shouldApplyQueueBulkActionToProject(project, action.projectId)
              ? project.queue.items.map((item) =>
                  isCancellableQueueItem(item) ? { ...item, status: 'cancelled' } : item
                )
              : project.queue.items,
          },
        })),
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
        projects: state.projects.map((project) => ({
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
        })),
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
      const normalizedServerProject = ensureProjectWidgetContracts(action.serverProject);
      const recoveredProject = ensureProjectWidgetContracts(action.recoveredProject);
      const localProject = state.projects.find((project) => project.id === action.projectId);
      const hasOriginal = localProject !== undefined;
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
      return addNotification(state, createNotification({ kind: 'error', message: action.message, title: 'Error' }));
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
    case 'refreshBackendData': {
      return { ...state, projects: state.projects.map((project) => refreshProjectBackendData(project)) };
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

export type { WorkbenchAction };
