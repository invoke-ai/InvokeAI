import {
  defaultInvocationRoute,
  isInvocationRouteValid,
  isInvocationSourceAvailable,
  resolveInvocationRoute,
} from './invocation';
import { compileGenerateGraph, resolveGenerateSeed } from './generation/graph';
import { normalizeGenerateWidgetValues } from './generation/settings';
import type { GallerySettings } from './gallery/settings';
import type { GenerateWidgetValues } from './generation/types';
import { defaultLayoutPreset, getLayoutPreset } from './layoutPresets';
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
import type { ProjectGraphState } from './workflows/types';
import type {
  CanvasDocumentContract,
  CanvasPlacementContract,
  CanvasStateContract,
  CenterViewId,
  CanvasRasterLayerContract,
  CanvasStagingCandidateContract,
  GeneratedImageContract,
  GraphContract,
  GraphHistorySnapshot,
  InvocationRoute,
  InvocationSourceId,
  LayoutPreset,
  LayoutPresetId,
  Project,
  ProjectLayoutState,
  ProjectSettings,
  ProjectUndoSnapshot,
  QueueItem,
  QueueItemStatus,
  ResultDestination,
  WidgetFailure,
  WidgetId,
  WidgetRegion,
  WidgetRegionState,
  WidgetStateContract,
  WorkbenchNotification,
  WorkbenchNotificationKind,
  WorkbenchState,
} from './types';

type WorkbenchAction =
  | { type: 'createProject' }
  | { type: 'openProject'; project: Project }
  | { type: 'closeProject'; projectId: string }
  | { type: 'renameProject'; projectId: string; name: string }
  | { type: 'switchProject'; projectId: string }
  | { type: 'setCenterView'; centerViewId: CenterViewId }
  | { type: 'applyPreset'; presetId: LayoutPresetId }
  | { type: 'resetActiveLayout' }
  | { type: 'recoverShellLayout' }
  | { type: 'setInvocationSource'; sourceId: InvocationSourceId }
  | { type: 'setInvocationDestination'; destination: ResultDestination }
  | { type: 'toggleSourceLock' }
  | { type: 'toggleDestinationLock' }
  | { type: 'openRegionWidget'; region: WidgetRegion; widgetId: WidgetId }
  | { type: 'selectRegionWidget'; region: WidgetRegion; widgetId: WidgetId }
  | { type: 'toggleRegionWidget'; region: WidgetRegion; widgetId: WidgetId }
  | { type: 'setRegionWidgetCollapsed'; region: WidgetRegion; isCollapsed: boolean }
  | { type: 'setRegionWidgetSize'; region: WidgetRegion; sizePx: number }
  | { type: 'setGenerateSettings'; values: GenerateWidgetValues }
  | { type: 'setGenerateBatchCount'; batchCount: number }
  | { type: 'patchWidgetValues'; widgetId: WidgetId; values: Record<string, unknown> }
  | { type: 'applyProjectGraphAction'; action: ProjectGraphAction }
  | { type: 'replaceProjectGraph'; document: ProjectGraphState; label: string }
  | { type: 'saveProjectGraphSnapshot' }
  | { type: 'restoreProjectGraphSnapshot'; snapshotId: string }
  | { type: 'setProjectGraphLibraryBinding'; libraryWorkflowId: string }
  | { type: 'submitInvocationSnapshot'; backendSupportsCancellation: boolean }
  | { type: 'submitResolvedInvocationSnapshot'; backendSupportsCancellation: boolean; route: InvocationRoute }
  | {
      type: 'markQueueItemBackendSubmitted';
      projectId: string;
      queueItemId: string;
      backendItemIds: number[];
      backendBatchId?: string;
    }
  | { type: 'setQueueItemStatus'; projectId: string; queueItemId: string; status: QueueItemStatus; error?: string }
  | { type: 'routeQueueItemResults'; projectId: string; queueItemId: string; images: GeneratedImageContract[] }
  | { type: 'setStagedImageIndex'; imageIndex: number }
  | { type: 'cycleStagedImage'; direction: -1 | 1 }
  | { type: 'discardSelectedStagedImage' }
  | { type: 'discardAllStagedImages' }
  | { type: 'toggleCanvasStagingVisibility' }
  | { type: 'toggleCanvasStagingThumbnailsVisibility' }
  | { type: 'selectGalleryImage'; image: GeneratedImageContract }
  | { type: 'toggleGalleryImageInSelection'; image: GeneratedImageContract }
  | { type: 'setGalleryMultiSelection'; imageNames: string[]; primaryImage: GeneratedImageContract }
  | { type: 'setGalleryCompareImage'; image: GeneratedImageContract | null }
  | { type: 'selectGalleryBoard'; boardId: string }
  | { type: 'setGalleryView'; galleryView: 'images' | 'assets' }
  | { type: 'setGallerySearchTerm'; searchTerm: string }
  | { type: 'updateGallerySettings'; settings: Partial<GallerySettings> }
  | { type: 'setGalleryPage'; page: number }
  | { type: 'setGalleryPageInfo'; totalImages: number }
  | { type: 'touchGalleryRefresh' }
  | { type: 'removeGalleryImages'; imageNames: string[] }
  | { type: 'setGalleryProjectBoardId'; boardId: string }
  | { type: 'acceptStagedImage' }
  | { type: 'clearCanvasStaging' }
  | { type: 'cancelQueueItem'; queueItemId: string }
  | { type: 'undoProjectChange' }
  | { type: 'redoProjectChange' }
  | { type: 'hydrateWorkbench'; state: WorkbenchState }
  | { type: 'reconcileProjectConflict'; projectId: string; serverProject: Project; recoveredProject: Project }
  | { type: 'autosaveStarted' }
  | { type: 'autosaveSucceeded'; savedAt: string }
  | { type: 'autosaveFailed'; error: string }
  | { type: 'markAllNotificationsRead' }
  | { type: 'clearNotifications' }
  | { type: 'clearErrorLog' }
  | { type: 'recordWidgetFailure'; failure: WidgetFailure }
  | { type: 'setActiveProjectSettings'; settings: Partial<ProjectSettings> }
  | { type: 'recordError'; message: string }
  | { type: 'setBackendConnectionStatus'; status: WorkbenchState['backendConnection']['status']; error?: string }
  | { type: 'refreshBackendData' }
  | { type: 'recordNotice'; kind: WorkbenchNotificationKind; title: string; message?: string };

const HISTORY_LIMIT = 40;
const ERROR_LOG_LIMIT = 5;
const NOTIFICATION_LIMIT = 100;
const MIN_PANEL_SIZE_PX = 180;
const MAX_PANEL_SIZE_PX = 520;
const MIN_STATUS_PANEL_SIZE_PX = 96;
const MAX_STATUS_PANEL_SIZE_PX = 420;
const DEFAULT_CANVAS_DOCUMENT_WIDTH = 1024;
const DEFAULT_CANVAS_DOCUMENT_HEIGHT = 1024;

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

const clonePlacement = (placement: CanvasPlacementContract): CanvasPlacementContract => ({ ...placement });

const createCenteredPlacement = (
  image: Pick<GeneratedImageContract, 'height' | 'width'>,
  document: Pick<CanvasDocumentContract, 'height' | 'width'>
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

const createCanvasDocument = (layers: CanvasRasterLayerContract[] = []): CanvasDocumentContract => ({
  height: DEFAULT_CANVAS_DOCUMENT_HEIGHT,
  layers,
  version: 1,
  width: DEFAULT_CANVAS_DOCUMENT_WIDTH,
});

const normalizeLayer = (layer: CanvasRasterLayerContract | string): CanvasRasterLayerContract => {
  if (typeof layer !== 'string') {
    return {
      ...layer,
      placement: layer.placement
        ? clonePlacement(layer.placement)
        : createCenteredPlacement(layer, createCanvasDocument()),
    };
  }

  return {
    acceptedAt: now(),
    height: 0,
    id: layer,
    imageName: layer,
    imageUrl: '',
    label: layer,
    placement: createCenteredPlacement({ height: 0, width: 0 }, createCanvasDocument()),
    queuedAt: now(),
    sourceQueueItemId: 'legacy',
    thumbnailUrl: '',
    width: 0,
  };
};

const normalizeCanvasDocument = (canvas: CanvasStateContract): CanvasDocumentContract => {
  const legacyCanvas = canvas as CanvasStateContract & { layers?: CanvasRasterLayerContract[] };
  const rawDocument = legacyCanvas.document;
  const document = rawDocument ?? createCanvasDocument(legacyCanvas.layers ?? []);

  return {
    height: document.height || DEFAULT_CANVAS_DOCUMENT_HEIGHT,
    layers: (document.layers ?? []).map(normalizeLayer),
    version: 1,
    width: document.width || DEFAULT_CANVAS_DOCUMENT_WIDTH,
  };
};

const normalizeStagingCandidate = (
  image: CanvasStagingCandidateContract | GeneratedImageContract,
  document: CanvasDocumentContract
): CanvasStagingCandidateContract => ({
  ...image,
  placement:
    'placement' in image && image.placement
      ? clonePlacement(image.placement)
      : createCenteredPlacement(image, document),
});

const clearStagingArea = (stagingArea: CanvasStateContract['stagingArea']): CanvasStateContract['stagingArea'] => ({
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

const getGalleryImages = (values: Record<string, unknown>): GeneratedImageContract[] =>
  Array.isArray(values.recentImages) ? (values.recentImages as GeneratedImageContract[]) : [];

const getGallerySelectedImageNames = (values: Record<string, unknown>): string[] => {
  if (Array.isArray(values.selectedImageNames)) {
    return (values.selectedImageNames as unknown[]).filter((name): name is string => typeof name === 'string');
  }

  return typeof values.selectedImageName === 'string' ? [values.selectedImageName] : [];
};

const cloneCanvas = (canvas: CanvasStateContract): CanvasStateContract => ({
  version: 1,
  document: normalizeCanvasDocument(canvas),
  stagingArea: {
    ...canvas.stagingArea,
    pendingImageIds: [...(canvas.stagingArea?.pendingImageIds ?? [])],
    pendingImages: (canvas.stagingArea?.pendingImages ?? []).map((image) =>
      normalizeStagingCandidate(image, normalizeCanvasDocument(canvas))
    ),
    areThumbnailsVisible: canvas.stagingArea?.areThumbnailsVisible ?? true,
    isVisible: canvas.stagingArea?.isVisible ?? (canvas.stagingArea?.pendingImages?.length ?? 0) > 0,
    selectedImageIndex: canvas.stagingArea?.selectedImageIndex ?? 0,
  },
});

const cloneWidgetState = (widgetState: WidgetStateContract): WidgetStateContract => ({
  ...widgetState,
  values: { ...widgetState.values },
});

const cloneWidgetStates = (
  widgetStates: Record<WidgetId, WidgetStateContract>
): Record<WidgetId, WidgetStateContract> => {
  const defaultWidgetStates = createWidgetStates();

  return {
    'autosave-status': cloneWidgetState(widgetStates['autosave-status'] ?? defaultWidgetStates['autosave-status']),
    canvas: cloneWidgetState(widgetStates.canvas ?? defaultWidgetStates.canvas),
    diagnostics: cloneWidgetState(widgetStates.diagnostics ?? defaultWidgetStates.diagnostics),
    gallery: cloneWidgetState(widgetStates.gallery ?? defaultWidgetStates.gallery),
    generate: cloneWidgetState(widgetStates.generate ?? defaultWidgetStates.generate),
    'history-controls': cloneWidgetState(widgetStates['history-controls'] ?? defaultWidgetStates['history-controls']),
    'layout-actions': cloneWidgetState(widgetStates['layout-actions'] ?? defaultWidgetStates['layout-actions']),
    layers: cloneWidgetState(widgetStates.layers ?? defaultWidgetStates.layers),
    models: cloneWidgetState(widgetStates.models ?? defaultWidgetStates.models),
    notifications: cloneWidgetState(widgetStates.notifications ?? defaultWidgetStates.notifications),
    preview: cloneWidgetState(widgetStates.preview ?? defaultWidgetStates.preview),
    project: cloneWidgetState(widgetStates.project ?? defaultWidgetStates.project),
    queue: cloneWidgetState(widgetStates.queue ?? defaultWidgetStates.queue),
    'server-status': cloneWidgetState(widgetStates['server-status'] ?? defaultWidgetStates['server-status']),
    users: cloneWidgetState(widgetStates.users ?? defaultWidgetStates.users),
    'version-status': cloneWidgetState(widgetStates['version-status'] ?? defaultWidgetStates['version-status']),
    workflow: cloneWidgetState(widgetStates.workflow ?? defaultWidgetStates.workflow),
  };
};

const cloneWidgetRegions = (
  widgetRegions: Record<WidgetRegion, WidgetRegionState>
): Record<WidgetRegion, WidgetRegionState> => ({
  center: {
    ...widgetRegions.center,
    enabledWidgetIds: [...widgetRegions.center.enabledWidgetIds],
  },
  left: {
    ...widgetRegions.left,
    enabledWidgetIds: [...widgetRegions.left.enabledWidgetIds],
  },
  right: {
    ...widgetRegions.right,
    enabledWidgetIds: [...widgetRegions.right.enabledWidgetIds],
  },
  bottom: {
    ...widgetRegions.bottom,
    enabledWidgetIds: [...widgetRegions.bottom.enabledWidgetIds],
  },
});

const cloneWidgetGraphs = (widgetGraphs: Project['widgetGraphs']): Project['widgetGraphs'] =>
  Object.fromEntries(Object.entries(widgetGraphs).map(([key, graph]) => [key, graph ? cloneGraph(graph) : graph]));

const createUndoSnapshot = (project: Project): ProjectUndoSnapshot => ({
  canvas: cloneCanvas(project.canvas),
  invocation: { ...project.invocation },
  layout: { ...project.layout, panels: { ...project.layout.panels } },
  projectGraph: cloneProjectGraph(project.projectGraph),
  widgetGraphs: cloneWidgetGraphs(project.widgetGraphs),
  widgetRegions: cloneWidgetRegions(project.widgetRegions),
  widgetStates: cloneWidgetStates(project.widgetStates),
});

const restoreUndoSnapshot = (project: Project, snapshot: ProjectUndoSnapshot): Project => ({
  ...project,
  canvas: { ...cloneCanvas(snapshot.canvas), stagingArea: cloneCanvas(project.canvas).stagingArea },
  invocation: { ...snapshot.invocation },
  layout: { ...snapshot.layout, panels: { ...snapshot.layout.panels } },
  projectGraph: cloneProjectGraph(normalizeProjectGraph(snapshot.projectGraph)),
  widgetGraphs: cloneWidgetGraphs(snapshot.widgetGraphs),
  widgetRegions: cloneWidgetRegions(snapshot.widgetRegions),
  widgetStates: cloneWidgetStates(snapshot.widgetStates),
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

const createWidgetStates = (): Record<WidgetId, WidgetStateContract> => ({
  'autosave-status': { id: 'autosave-status', label: 'Autosave', values: {}, version: 1 },
  canvas: { id: 'canvas', label: 'Canvas', values: {}, version: 1 },
  diagnostics: { id: 'diagnostics', label: 'Diagnostics', values: {}, version: 1 },
  gallery: { id: 'gallery', label: 'Gallery', values: {}, version: 1 },
  generate: { graphId: 'generate-graph', id: 'generate', label: 'Generate', values: {}, version: 1 },
  'history-controls': { id: 'history-controls', label: 'History Controls', values: {}, version: 1 },
  'layout-actions': { id: 'layout-actions', label: 'Layout Actions', values: {}, version: 1 },
  layers: { id: 'layers', label: 'Layers', values: {}, version: 1 },
  models: { id: 'models', label: 'Models', values: {}, version: 1 },
  notifications: { id: 'notifications', label: 'Notifications', values: {}, version: 1 },
  preview: { id: 'preview', label: 'Preview', values: {}, version: 1 },
  project: { id: 'project', label: 'Project', values: {}, version: 1 },
  queue: { id: 'queue', label: 'Queue', values: {}, version: 1 },
  'server-status': { id: 'server-status', label: 'Server Status', values: {}, version: 1 },
  users: { id: 'users', label: 'Users', values: {}, version: 1 },
  'version-status': { id: 'version-status', label: 'Version', values: {}, version: 1 },
  workflow: { graphId: 'workflow-graph', id: 'workflow', label: 'Workflow', values: {}, version: 1 },
});

const createWidgetRegions = (): Record<WidgetRegion, WidgetRegionState> => ({
  left: {
    activeWidgetId: 'generate',
    enabledWidgetIds: ['generate', 'workflow'],
    isCollapsed: false,
    sizePx: 288,
  },
  right: {
    activeWidgetId: 'layers',
    enabledWidgetIds: ['queue', 'gallery', 'layers', 'models', 'diagnostics', 'project'],
    isCollapsed: false,
    sizePx: 240,
  },
  bottom: {
    activeWidgetId: 'queue',
    enabledWidgetIds: [
      'server-status',
      'diagnostics',
      'queue',
      'gallery',
      'notifications',
      'autosave-status',
      'history-controls',
      'layout-actions',
      'version-status',
      'workflow',
    ],
    isCollapsed: true,
    sizePx: 180,
  },
  center: {
    activeWidgetId: 'canvas',
    enabledWidgetIds: ['canvas', 'gallery', 'preview', 'workflow', 'models'],
    isCollapsed: false,
    sizePx: 0,
  },
});

const LEGACY_RIGHT_REGION_WIDGET_IDS: WidgetId[] = ['queue', 'gallery', 'layers'];

const isLegacyDefaultRightRegion = (region: WidgetRegionState): boolean =>
  region.enabledWidgetIds.length === LEGACY_RIGHT_REGION_WIDGET_IDS.length &&
  region.enabledWidgetIds.every((widgetId, index) => widgetId === LEGACY_RIGHT_REGION_WIDGET_IDS[index]);

const ensureRightRegion = (rightRegion: WidgetRegionState | undefined): WidgetRegionState => {
  const defaultRightRegion = createWidgetRegions().right;

  if (!rightRegion) {
    return defaultRightRegion;
  }

  if (isLegacyDefaultRightRegion(rightRegion)) {
    return { ...rightRegion, enabledWidgetIds: defaultRightRegion.enabledWidgetIds };
  }

  return rightRegion;
};

const getCenterWidgetIdFromViewId = (centerViewId: CenterViewId): WidgetId => centerViewId;

const ensureCenterRegion = (
  centerRegion: WidgetRegionState | undefined,
  fallbackCenterViewId: CenterViewId
): WidgetRegionState => {
  const defaultCenterRegion = createWidgetRegions().center;
  const activeWidgetId = centerRegion?.activeWidgetId ?? getCenterWidgetIdFromViewId(fallbackCenterViewId);
  const enabledWidgetIds = centerRegion?.enabledWidgetIds.length
    ? centerRegion.enabledWidgetIds
    : defaultCenterRegion.enabledWidgetIds;
  const normalizedActiveWidgetId = enabledWidgetIds.includes(activeWidgetId) ? activeWidgetId : enabledWidgetIds[0];

  return {
    ...defaultCenterRegion,
    ...centerRegion,
    activeWidgetId: normalizedActiveWidgetId,
    enabledWidgetIds,
    isCollapsed: false,
  };
};

const ensureProjectWidgetContracts = (project: Project): Project => {
  const defaultWidgetRegions = createWidgetRegions();
  const defaultWidgetStates = createWidgetStates();
  const legacyWidgetRegions = project.widgetRegions as
    | Partial<Record<WidgetRegion | 'left-panel' | 'right-panel' | 'status-bar', WidgetRegionState>>
    | undefined;

  return {
    ...project,
    canvas: cloneCanvas(project.canvas ?? createCanvasState()),
    projectGraph: normalizeProjectGraph(project.projectGraph),
    settings: normalizeProjectSettings(project.settings),
    widgetRegions: {
      left: legacyWidgetRegions?.left ?? legacyWidgetRegions?.['left-panel'] ?? defaultWidgetRegions.left,
      right: ensureRightRegion(legacyWidgetRegions?.right ?? legacyWidgetRegions?.['right-panel']),
      bottom: legacyWidgetRegions?.bottom ?? legacyWidgetRegions?.['status-bar'] ?? defaultWidgetRegions.bottom,
      center: ensureCenterRegion(legacyWidgetRegions?.center, project.layout.centerViewId),
    },
    widgetStates: {
      ...defaultWidgetStates,
      ...project.widgetStates,
    },
  };
};

const clampPanelSize = (region: WidgetRegion, sizePx: number): number => {
  if (region === 'bottom') {
    return Math.min(MAX_STATUS_PANEL_SIZE_PX, Math.max(MIN_STATUS_PANEL_SIZE_PX, sizePx));
  }

  return Math.min(MAX_PANEL_SIZE_PX, Math.max(MIN_PANEL_SIZE_PX, sizePx));
};

const createCanvasState = (): CanvasStateContract => ({
  document: createCanvasDocument(),
  stagingArea: {
    areThumbnailsVisible: true,
    isVisible: false,
    pendingImageIds: [],
    pendingImages: [],
    selectedImageIndex: 0,
  },
  version: 1,
});

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
  layout: { ...defaultLayoutPreset.initialLayout, panels: { ...defaultLayoutPreset.initialLayout.panels } },
  name: `Project Name #${index}`,
  projectGraph: createProjectGraph(`${id}-graph`),
  queue: { items: [] },
  settings: normalizeProjectSettings(),
  undoRedo: { future: [], past: [] },
  widgetGraphs: {},
  widgetRegions: createWidgetRegions(),
  widgetStates: createWidgetStates(),
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

const updateActiveProject = (state: WorkbenchState, getProject: (project: Project) => Project): WorkbenchState => ({
  ...state,
  projects: state.projects.map((project) =>
    project.id === state.activeProjectId ? getProject(ensureProjectWidgetContracts(project)) : project
  ),
});

const getNextEnabledWidgetId = (region: WidgetRegionState, widgetId: WidgetId): WidgetId | null => {
  if (region.activeWidgetId !== widgetId) {
    return region.activeWidgetId;
  }

  return region.enabledWidgetIds.find((enabledWidgetId) => enabledWidgetId !== widgetId) ?? null;
};

const updateActiveWidgetRegion = (
  state: WorkbenchState,
  region: WidgetRegion,
  getRegion: (regionState: WidgetRegionState) => WidgetRegionState
): WorkbenchState =>
  updateActiveProject(state, (project) => ({
    ...project,
    widgetRegions: {
      ...project.widgetRegions,
      [region]: getRegion(project.widgetRegions[region]),
    },
  }));

/** Focuses the regions a layout preset names: the center view and, when declared, a left-rail widget. */
const applyPresetRegionFocus = (state: WorkbenchState, preset: LayoutPreset): WorkbenchState => {
  const centerWidgetId = getCenterWidgetIdFromViewId(preset.initialLayout.centerViewId);
  const nextState = updateActiveWidgetRegion(state, 'center', (region) => ({
    ...region,
    activeWidgetId: region.enabledWidgetIds.includes(centerWidgetId) ? centerWidgetId : region.activeWidgetId,
    isCollapsed: false,
  }));
  const leftWidgetId = preset.leftRegionWidgetId;

  if (!leftWidgetId) {
    return nextState;
  }

  return updateActiveWidgetRegion(nextState, 'left', (region) => ({
    ...region,
    activeWidgetId: region.enabledWidgetIds.includes(leftWidgetId) ? leftWidgetId : region.activeWidgetId,
    isCollapsed: false,
  }));
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

const normalizeWorkbenchState = (state: WorkbenchState): WorkbenchState => ({
  ...state,
  backendConnection: { status: 'connecting' },
  // Built explicitly: legacy snapshots carried preferences inside the account
  // (they live in the settings store now) and must not resurface here.
  account: { activeLayoutPresetId: state.account?.activeLayoutPresetId ?? defaultLayoutPreset.id },
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
  route: InvocationRoute
): { graph: GraphContract; widgetStates: Record<WidgetId, WidgetStateContract> } | null => {
  const widgetStates = cloneWidgetStates(project.widgetStates);

  if (route.sourceId === 'project-graph') {
    // Compiles the workflow document into an immutable snapshot. Templates are
    // read imperatively; route validation already guaranteed they are loaded.
    const templatesSnapshot = getInvocationTemplatesSnapshot();

    if (templatesSnapshot.status !== 'loaded') {
      return null;
    }

    return { graph: compileProjectGraph(project.projectGraph, templatesSnapshot.templates), widgetStates };
  }

  if (route.sourceId !== 'generate') {
    const widgetGraph = project.widgetGraphs[route.sourceId as WidgetId];

    return widgetGraph ? { graph: cloneGraph(widgetGraph), widgetStates } : null;
  }

  const values = normalizeGenerateWidgetValues(project.widgetStates.generate.values);

  if (!values) {
    return null;
  }

  const resolvedSettings: GenerateWidgetValues = {
    ...values,
    seed: resolveGenerateSeed(values),
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
    values: { ...resolvedSettings, model: { ...resolvedSettings.model } },
  };

  return { graph: compiledGraph, widgetStates };
};

const updateProjectById = (
  state: WorkbenchState,
  projectId: string,
  getProject: (project: Project) => Project
): WorkbenchState => ({
  ...state,
  projects: state.projects.map((project) =>
    project.id === projectId ? getProject(ensureProjectWidgetContracts(project)) : project
  ),
});

const updateGalleryValues = (
  state: WorkbenchState,
  getValues: (values: Record<string, unknown>) => Record<string, unknown>
): WorkbenchState =>
  updateActiveProject(state, (project) => ({
    ...project,
    widgetStates: {
      ...project.widgetStates,
      gallery: {
        ...project.widgetStates.gallery,
        values: getValues(project.widgetStates.gallery.values),
      },
    },
  }));

const refreshProjectBackendData = (project: Project): Project => ({
  ...project,
  widgetStates: {
    ...project.widgetStates,
    gallery: {
      ...project.widgetStates.gallery,
      values: {
        ...project.widgetStates.gallery.values,
        galleryRefreshToken: createId('gallery-refresh'),
      },
    },
  },
});

const updateQueueItem = (project: Project, queueItemId: string, getItem: (item: QueueItem) => QueueItem): Project => ({
  ...project,
  queue: {
    items: project.queue.items.map((item) => (item.id === queueItemId ? getItem(item) : item)),
  },
});

const routeQueueItemResults = (project: Project, queueItemId: string, images: GeneratedImageContract[]): Project => {
  const queueItem = project.queue.items.find((item) => item.id === queueItemId);
  const destination = queueItem?.snapshot.destination ?? project.invocation.destination;
  const nextProject = updateQueueItem(project, queueItemId, (item) => ({
    ...item,
    resultImages: images,
    status: 'completed',
  }));

  if (destination === 'gallery') {
    const galleryValues = nextProject.widgetStates.gallery.values;
    const existingImages = getGalleryImages(galleryValues).filter(
      (image) => !images.some((incomingImage) => incomingImage.imageName === image.imageName)
    );

    return {
      ...nextProject,
      widgetStates: {
        ...nextProject.widgetStates,
        gallery: {
          ...nextProject.widgetStates.gallery,
          values: {
            ...galleryValues,
            recentImages: [...images, ...existingImages],
            selectedImage: images[0] ?? galleryValues.selectedImage,
            selectedImageName: images[0]?.imageName ?? nextProject.widgetStates.gallery.values.selectedImageName,
            selectedImageNames: images[0] ? [images[0].imageName] : getGallerySelectedImageNames(galleryValues),
          },
        },
      },
    };
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
        selectedImageIndex:
          incomingImages.length > 0 ? existingImages.length : nextProject.canvas.stagingArea.selectedImageIndex,
        sourceQueueItemId: queueItemId,
      },
    },
  };
};

const submitInvocationSnapshot = (
  project: Project,
  backendSupportsCancellation: boolean,
  route = resolveInvocationRoute(project)
): Project => {
  if (!isInvocationRouteValid(route)) {
    return project;
  }

  const submittedAt = now();
  const queueItemId = createId('queue-item');
  const compiledSnapshot = compileInvocationSnapshot(project, route);

  if (!compiledSnapshot) {
    return project;
  }

  const { graph, widgetStates } = compiledSnapshot;
  const graphHistorySnapshot = createGraphHistorySnapshot(`Queue snapshot ${queueItemId}`, graph);
  const queueItem: QueueItem = {
    cancellable: backendSupportsCancellation,
    id: queueItemId,
    snapshot: {
      canvas: cloneCanvas(project.canvas),
      destination: route.destination,
      graph,
      sourceId: route.sourceId,
      submittedAt,
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

export const createInitialWorkbenchState = (): WorkbenchState => {
  const draft = createDraftProject([]);

  return {
    account: { activeLayoutPresetId: defaultLayoutPreset.id },
    activeProjectId: draft.id,
    autosave: { status: 'idle' },
    backendConnection: { status: 'connecting' },
    errorLog: [],
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

        return addNotification(
          {
            ...state,
            errorLog: [message, ...state.errorLog],
          },
          createNotification({ kind: 'error', message, title: 'Project close blocked' })
        );
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
        activeWidgetId: region.enabledWidgetIds.includes(widgetId) ? widgetId : region.activeWidgetId,
        isCollapsed: false,
      }));
    }
    case 'applyPreset': {
      const preset = getLayoutPreset(action.presetId);
      const nextState = updateActiveLayout(state, () => ({
        ...preset.initialLayout,
        panels: { ...preset.initialLayout.panels },
      }));

      return {
        ...applyPresetRegionFocus(nextState, preset),
        account: { ...state.account, activeLayoutPresetId: action.presetId },
      };
    }
    case 'resetActiveLayout': {
      const preset = getLayoutPreset(
        state.projects.find((project) => project.id === state.activeProjectId)?.layout.presetId ??
          state.account.activeLayoutPresetId
      );
      const nextState = updateActiveLayout(state, () => {
        return { ...preset.initialLayout, panels: { ...preset.initialLayout.panels } };
      });

      return applyPresetRegionFocus(nextState, preset);
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
      return updateActiveProject(state, (project) => {
        const region = project.widgetRegions[action.region];
        const enabledWidgetIds = region.enabledWidgetIds.includes(action.widgetId)
          ? region.enabledWidgetIds
          : [...region.enabledWidgetIds, action.widgetId];

        return {
          ...project,
          layout: openPanelForRegion(project.layout, action.region),
          widgetRegions: {
            ...project.widgetRegions,
            [action.region]: {
              ...region,
              activeWidgetId: action.widgetId,
              enabledWidgetIds,
              isCollapsed: false,
            },
          },
        };
      });
    }
    case 'selectRegionWidget': {
      return updateActiveProject(state, (project) => {
        const region = project.widgetRegions[action.region];

        if (action.region === 'center') {
          return {
            ...project,
            widgetRegions: {
              ...project.widgetRegions,
              center: { ...region, activeWidgetId: action.widgetId, isCollapsed: false },
            },
          };
        }

        const widgetRegion =
          region.activeWidgetId === action.widgetId
            ? { ...region, isCollapsed: !region.isCollapsed }
            : { ...region, activeWidgetId: action.widgetId, isCollapsed: false };

        return {
          ...project,
          layout: openPanelForRegion(project.layout, action.region),
          widgetRegions: { ...project.widgetRegions, [action.region]: widgetRegion },
        };
      });
    }
    case 'toggleRegionWidget': {
      return updateActiveWidgetRegion(state, action.region, (region) => {
        const isEnabled = region.enabledWidgetIds.includes(action.widgetId);

        if (action.region === 'center' && isEnabled && region.enabledWidgetIds.length === 1) {
          return region;
        }

        const enabledWidgetIds = isEnabled
          ? region.enabledWidgetIds.filter((widgetId) => widgetId !== action.widgetId)
          : [...region.enabledWidgetIds, action.widgetId];
        const fallbackWidgetId = getNextEnabledWidgetId(region, action.widgetId);

        return {
          ...region,
          activeWidgetId: isEnabled && fallbackWidgetId ? fallbackWidgetId : action.widgetId,
          enabledWidgetIds,
          isCollapsed: action.region === 'center' ? false : enabledWidgetIds.length === 0 ? true : region.isCollapsed,
        };
      });
    }
    case 'setRegionWidgetCollapsed': {
      if (action.region === 'center') {
        return state;
      }

      return updateActiveWidgetRegion(state, action.region, (region) => ({
        ...region,
        isCollapsed: action.isCollapsed,
      }));
    }
    case 'setRegionWidgetSize': {
      return updateActiveWidgetRegion(state, action.region, (region) => ({
        ...region,
        sizePx: clampPanelSize(action.region, action.sizePx),
      }));
    }
    case 'setGenerateSettings': {
      return updateActiveProject(state, (project) => ({
        ...project,
        widgetStates: {
          ...project.widgetStates,
          generate: {
            ...project.widgetStates.generate,
            values: { ...action.values, model: { ...action.values.model } },
          },
        },
      }));
    }
    case 'setGenerateBatchCount': {
      const batchCount = Math.min(64, Math.max(1, Math.round(action.batchCount)));

      return updateActiveProject(state, (project) => ({
        ...project,
        widgetStates: {
          ...project.widgetStates,
          generate: {
            ...project.widgetStates.generate,
            values: { ...project.widgetStates.generate.values, batchCount },
          },
        },
      }));
    }
    case 'patchWidgetValues': {
      // Generic widget-owned UI state (panel modes, tabs, sizes). Not undoable.
      return updateActiveProject(state, (project) => ({
        ...project,
        widgetStates: {
          ...project.widgetStates,
          [action.widgetId]: {
            ...project.widgetStates[action.widgetId],
            values: { ...project.widgetStates[action.widgetId].values, ...action.values },
          },
        },
      }));
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
          project.invocation.sourceId !== 'project-graph' &&
          isHighConfidenceGraphEdit(action.action) &&
          isInvocationSourceAvailable('project-graph');

        return {
          ...nextProject,
          invocation: shouldAutoSetSource
            ? { ...nextProject.invocation, sourceId: 'project-graph' }
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
        submitInvocationSnapshot(project, action.backendSupportsCancellation)
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
          resolveInvocationRoute(project, 'global', action.route)
        )
      );
    }
    case 'markQueueItemBackendSubmitted': {
      return updateProjectById(state, action.projectId, (project) =>
        updateQueueItem(project, action.queueItemId, (item) => ({
          ...item,
          backendBatchId: action.backendBatchId,
          backendItemIds: action.backendItemIds,
          status: item.status === 'cancelled' ? 'cancelled' : 'running',
        }))
      );
    }
    case 'setQueueItemStatus': {
      const project = state.projects.find((project) => project.id === action.projectId);
      const queueItem = project?.queue.items.find((item) => item.id === action.queueItemId);

      if (queueItem?.status === 'cancelled' && action.status !== 'cancelled') {
        return state;
      }

      const nextState = updateProjectById(state, action.projectId, (project) =>
        updateQueueItem(project, action.queueItemId, (item) => ({
          ...item,
          error: action.error,
          status: action.status,
        }))
      );

      if (action.status !== 'failed' && action.status !== 'cancelled') {
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
      return updateGalleryValues(state, (values) => ({
        ...values,
        selectedImage: action.image,
        selectedImageName: action.image.imageName,
        selectedImageNames: [action.image.imageName],
      }));
    }
    case 'toggleGalleryImageInSelection': {
      return updateGalleryValues(state, (values) => {
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
      });
    }
    case 'setGalleryMultiSelection': {
      return updateGalleryValues(state, (values) => ({
        ...values,
        selectedImage: action.primaryImage,
        selectedImageName: action.primaryImage.imageName,
        selectedImageNames: action.imageNames,
      }));
    }
    case 'setGalleryCompareImage': {
      return updateGalleryValues(state, (values) => ({ ...values, compareImage: action.image }));
    }
    case 'selectGalleryBoard': {
      return updateGalleryValues(state, (values) => ({
        ...values,
        galleryPage: 0,
        selectedBoardId: action.boardId,
        selectedImageNames: [],
      }));
    }
    case 'setGalleryView': {
      return updateGalleryValues(state, (values) => ({
        ...values,
        galleryPage: 0,
        galleryView: action.galleryView,
        selectedImageNames: [],
      }));
    }
    case 'setGallerySearchTerm': {
      return updateGalleryValues(state, (values) => ({
        ...values,
        galleryPage: 0,
        searchTerm: action.searchTerm,
      }));
    }
    case 'updateGallerySettings': {
      const resetsQuery =
        action.settings.imageOrderDir !== undefined ||
        action.settings.starredFirst !== undefined ||
        action.settings.paginationMode !== undefined;

      return updateGalleryValues(state, (values) => ({
        ...values,
        ...action.settings,
        ...(resetsQuery ? { galleryPage: 0 } : {}),
      }));
    }
    case 'setGalleryPage': {
      return updateGalleryValues(state, (values) => ({ ...values, galleryPage: Math.max(0, action.page) }));
    }
    case 'setGalleryPageInfo': {
      return updateGalleryValues(state, (values) => ({ ...values, galleryTotalImages: action.totalImages }));
    }
    case 'touchGalleryRefresh': {
      return updateGalleryValues(state, (values) => ({ ...values, galleryRefreshToken: createId('gallery-refresh') }));
    }
    case 'removeGalleryImages': {
      const removedImageNames = new Set(action.imageNames);

      return updateGalleryValues(state, (values) => {
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
      });
    }
    case 'setGalleryProjectBoardId': {
      return updateGalleryValues(state, (values) => ({ ...values, projectBoardId: action.boardId }));
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
        const nextProject = pushUndo(project, 'Accept staged canvas candidate');
        const layer: CanvasRasterLayerContract = {
          ...stagedImage,
          acceptedAt,
          id: createId('layer'),
          label: `Layer ${project.canvas.document.layers.length + 1}`,
          placement: clonePlacement(stagedImage.placement),
        };

        return {
          ...nextProject,
          canvas: {
            ...nextProject.canvas,
            document: {
              ...nextProject.canvas.document,
              layers: [layer, ...nextProject.canvas.document.layers],
            },
            stagingArea: clearStagingArea(nextProject.canvas.stagingArea),
          },
          events: [
            {
              createdAt: acceptedAt,
              id: createId('event'),
              summary: `Accepted ${stagedImage.imageName} into a new raster layer`,
              type: 'canvas-layer-accepted',
            },
            ...nextProject.events,
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
    case 'cancelQueueItem': {
      const activeProject = state.projects.find((project) => project.id === state.activeProjectId);
      const queueItem = activeProject?.queue.items.find((item) => item.id === action.queueItemId);
      const canCancelQueueItem =
        Boolean(queueItem?.cancellable) && (queueItem?.status === 'pending' || queueItem?.status === 'running');
      const nextState = updateActiveProject(state, (project) => ({
        ...project,
        queue: {
          items: project.queue.items.map((item) => {
            if (
              item.id !== action.queueItemId ||
              !item.cancellable ||
              (item.status !== 'pending' && item.status !== 'running')
            ) {
              return item;
            }

            return { ...item, status: 'cancelled' };
          }),
        },
      }));

      if (!activeProject || !queueItem || !canCancelQueueItem) {
        return nextState;
      }

      return addNotification(
        nextState,
        createNotification({
          kind: 'info',
          message: `${activeProject.name}: ${action.queueItemId}`,
          projectId: activeProject.id,
          title: 'Invocation cancellation requested',
        })
      );
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
      const serverProject = ensureProjectWidgetContracts(action.serverProject);
      const recoveredProject = ensureProjectWidgetContracts(action.recoveredProject);
      const hasOriginal = state.projects.some((project) => project.id === action.projectId);
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
    case 'clearErrorLog': {
      return { ...state, errorLog: [] };
    }
    case 'recordWidgetFailure': {
      const hasFailure = state.widgetFailures.some((failure) => failure.widgetId === action.failure.widgetId);

      if (hasFailure) {
        return state;
      }

      return addNotification(
        {
          ...state,
          errorLog: [action.failure.details, ...state.errorLog].slice(0, ERROR_LOG_LIMIT),
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
      return addNotification(
        { ...state, errorLog: [action.message, ...state.errorLog].slice(0, ERROR_LOG_LIMIT) },
        createNotification({ kind: 'error', message: action.message, title: 'Error' })
      );
    }
    case 'setBackendConnectionStatus': {
      const timestamp = now();

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
      return updateActiveProject(state, (project) => ({
        ...project,
        settings: normalizeProjectSettings({ ...project.settings, ...action.settings }),
      }));
    }
  }
};

export type { WorkbenchAction };
