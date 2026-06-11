import type { ComponentType } from 'react';

export type LayoutPresetId = 'canvas-default' | 'gallery' | 'workflow' | 'linear' | 'model-manager';

export type CenterViewId = 'canvas' | 'gallery' | 'preview' | 'workflow' | 'models';

export type GraphId = string;

export type WidgetId =
  | 'autosave-status'
  | 'canvas'
  | 'diagnostics'
  | 'gallery'
  | 'generate'
  | 'history-controls'
  | 'layout-actions'
  | 'layers'
  | 'models'
  | 'notifications'
  | 'preview'
  | 'project'
  | 'queue'
  | 'server-status'
  | 'users'
  | 'version-status'
  | 'workflow';

export type WorkbenchRegion = 'left' | 'right' | 'center' | 'bottom' | 'dialog' | 'popover';

export type QueueItemStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export type ProjectEventType =
  | 'project-created'
  | 'layout-updated'
  | 'invocation-updated'
  | 'queue-submitted'
  | 'canvas-layer-accepted';

export interface GraphNodeContract {
  id: string;
  type: string;
  inputs: Record<string, unknown>;
}

export interface GraphEdgeContract {
  id: string;
  sourceNodeId: string;
  sourceField: string;
  targetNodeId: string;
  targetField: string;
}

export interface GraphContract {
  id: GraphId;
  version: 1;
  label: string;
  nodes: GraphNodeContract[];
  edges: GraphEdgeContract[];
  updatedAt: string;
  backendGraph?: BackendGraphContract;
}

export interface BackendInvocationContract {
  id: string;
  type: string;
  [key: string]: unknown;
}

export interface BackendGraphEdgeContract {
  source: {
    node_id: string;
    field: string;
  };
  destination: {
    node_id: string;
    field: string;
  };
}

export interface BackendGraphContract {
  id: string;
  nodes: Record<string, BackendInvocationContract>;
  edges: BackendGraphEdgeContract[];
}

export interface WidgetStateContract {
  id: WidgetId;
  label: string;
  version: 1;
  values: Record<string, unknown>;
  graphId?: GraphId;
}

export interface GraphBearingSurfaceContract {
  surfaceId: string;
  widgetId: WidgetId;
  label: string;
  sourceId: InvocationSourceId;
  graphId: GraphId;
  region: WorkbenchRegion;
  canSetSource: boolean;
  canPreviewGraph: boolean;
}

export interface WidgetViewProps {
  region: WorkbenchRegion;
  manifest: WidgetManifest;
  presentation?: 'compact' | 'expanded' | 'tooltip';
}

export type WidgetView = ComponentType<WidgetViewProps>;

export type WidgetHeaderActions = ComponentType<WidgetViewProps>;
export type WidgetFooter = ComponentType<WidgetViewProps>;

export interface WidgetLabelProps {
  region: WorkbenchRegion;
  presentation?: 'compact' | 'expanded' | 'tooltip';
}

export type WidgetLabel = string | ComponentType<WidgetLabelProps>;

export type WidgetIconId = `lucide-react:${string}`;

export interface WidgetManifest {
  id: WidgetId;
  label: WidgetLabel;
  labelText: string;
  version: 1;
  regions: WorkbenchRegion[];
  icon: WidgetIconId;
  bottomPanel?: 'expandable' | 'tooltip';
  centerPlacement?: 'toolbar' | 'view';
  /** Only offered while an admin is signed in to a multi-user backend. */
  requiresAdmin?: boolean;
  chrome?: {
    header?: 'hidden' | 'visible';
  };
  view?: WidgetView;
  headerActions?: WidgetHeaderActions;
  footer?: WidgetFooter;
  graphBearing?: {
    sourceId: InvocationSourceId;
    defaultGraphId: GraphId;
    surfaces: WorkbenchRegion[];
  };
  failurePolicy: {
    onRegistrationFailure: 'disable' | 'hide';
    isolateRenderFailure: boolean;
  };
}

export interface RegisteredWidget {
  manifest: WidgetManifest;
  status: 'enabled' | 'disabled' | 'hidden';
  failure?: WidgetFailure;
}

export interface WidgetFailure {
  widgetId: WidgetId;
  message: string;
  details: string;
  occurredAt: string;
}

export interface GeneratedImageContract {
  imageName: string;
  imageUrl: string;
  thumbnailUrl: string;
  width: number;
  height: number;
  queuedAt: string;
  sourceQueueItemId: string;
}

export interface CanvasPlacementContract {
  x: number;
  y: number;
  width: number;
  height: number;
  opacity: number;
}

export interface CanvasStagingCandidateContract extends GeneratedImageContract {
  placement: CanvasPlacementContract;
}

export interface CanvasRasterLayerContract extends GeneratedImageContract {
  id: string;
  acceptedAt: string;
  label: string;
  placement: CanvasPlacementContract;
}

export interface CanvasDocumentContract {
  version: 1;
  width: number;
  height: number;
  layers: CanvasRasterLayerContract[];
}

export interface CanvasStateContract {
  version: 1;
  document: CanvasDocumentContract;
  stagingArea: {
    sourceQueueItemId?: string;
    selectedLayerId?: string;
    pendingImageIds: string[];
    pendingImages: CanvasStagingCandidateContract[];
    selectedImageIndex: number;
    isVisible: boolean;
    areThumbnailsVisible: boolean;
  };
}

/**
 * Invocation sources and destinations.
 *
 * Phase 1 only needs enough of the Invocation Controller to render a stable,
 * fixed-width `Source → Destination` route on the global Invoke control. The
 * full resolver (auto vs. locked, validation, dialog focus) arrives in Phase 4,
 * but the project-owned shape is modelled now so the control is not a dead
 * placeholder.
 */
export type InvocationSourceId = 'generate' | 'project-graph' | 'upscale' | 'canvas-fill';

export type InvocationMode = 'global' | 'dialog';

export type ResultDestination = 'canvas' | 'gallery';

export interface InvocationRoute {
  sourceId: InvocationSourceId;
  destination: ResultDestination;
  sourceLocked: boolean;
  destinationLocked: boolean;
}

export interface ResolvedInvocationRoute extends InvocationRoute {
  mode: InvocationMode;
  sourceValid: boolean;
  destinationValid: boolean;
  validationMessage?: string;
}

export interface InvocationControllerState extends InvocationRoute {
  lastSubmittedRunId?: string;
}

export interface PanelState {
  isLeftOpen: boolean;
  isRightOpen: boolean;
  isBottomOpen: boolean;
}

export type WidgetRegion = 'left' | 'right' | 'bottom' | 'center';

export interface WidgetRegionState {
  activeWidgetId: WidgetId;
  enabledWidgetIds: WidgetId[];
  isCollapsed: boolean;
  sizePx: number;
}

export interface ProjectLayoutState {
  presetId: LayoutPresetId;
  centerViewId: CenterViewId;
  panels: PanelState;
}

export interface Project {
  id: string;
  name: string;
  /**
   * Set on recovery forks created when a save loses a revision race: the id
   * of the root project this recovered from (chains collapse to the root, so
   * a recovery of a recovery still points at the original).
   */
  recoveryOf?: string;
  recoveredAt?: string;
  layout: ProjectLayoutState;
  invocation: InvocationControllerState;
  projectGraph: GraphContract;
  widgetStates: Record<WidgetId, WidgetStateContract>;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
  widgetGraphs: Partial<Record<WidgetId, GraphContract>>;
  canvas: CanvasStateContract;
  graphHistory: GraphHistorySnapshot[];
  undoRedo: UndoRedoHistory;
  queue: QueueState;
  events: ProjectEvent[];
}

export interface LayoutPreset {
  id: LayoutPresetId;
  label: string;
  description: string;
  initialLayout: ProjectLayoutState;
}

export interface WorkbenchState {
  projects: Project[];
  activeProjectId: string;
  backendConnection: BackendConnectionState;
  errorLog: string[];
  notifications: WorkbenchNotification[];
  autosave: AutosaveState;
  account: AccountState;
  widgetFailures: WidgetFailure[];
}

export type BackendConnectionStatus = 'connecting' | 'connected' | 'disconnected';

export interface BackendConnectionState {
  status: BackendConnectionStatus;
  error?: string;
  lastConnectedAt?: string;
  lastDisconnectedAt?: string;
}

export type WorkbenchNotificationKind = 'error' | 'success' | 'info';

export interface WorkbenchNotification {
  id: string;
  kind: WorkbenchNotificationKind;
  title: string;
  message?: string;
  createdAt: string;
  projectId?: string;
  isRead: boolean;
}

export interface GraphHistorySnapshot {
  id: string;
  createdAt: string;
  label: string;
  graph: GraphContract;
}

export interface UndoRedoEntry {
  id: string;
  createdAt: string;
  label: string;
  project: ProjectUndoSnapshot;
}

export interface ProjectUndoSnapshot {
  layout: ProjectLayoutState;
  invocation: InvocationControllerState;
  projectGraph: GraphContract;
  widgetStates: Record<WidgetId, WidgetStateContract>;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
  widgetGraphs: Partial<Record<WidgetId, GraphContract>>;
  canvas: CanvasStateContract;
}

export interface UndoRedoHistory {
  past: UndoRedoEntry[];
  future: UndoRedoEntry[];
}

export interface QueueSubmissionSnapshot {
  sourceId: InvocationSourceId;
  destination: ResultDestination;
  graph: GraphContract;
  widgetStates: Record<WidgetId, WidgetStateContract>;
  canvas: CanvasStateContract;
  submittedAt: string;
}

export interface QueueItem {
  id: string;
  status: QueueItemStatus;
  cancellable: boolean;
  snapshot: QueueSubmissionSnapshot;
  backendItemIds?: number[];
  backendBatchId?: string;
  error?: string;
  resultImages?: GeneratedImageContract[];
}

export interface QueueState {
  items: QueueItem[];
}

export interface ProjectEvent {
  id: string;
  type: ProjectEventType;
  createdAt: string;
  summary: string;
  runId?: string;
}

export interface RunRecord {
  id: string;
  projectId: string;
  queueItemId: string;
  sourceId: InvocationSourceId;
  destination: ResultDestination;
  graphSnapshotId: string;
  status: QueueItemStatus;
  submittedAt: string;
  completedAt?: string;
}

export interface AutosaveState {
  status: 'idle' | 'saving' | 'saved' | 'error';
  lastSavedAt?: string;
  error?: string;
}

/**
 * Identifier for a workbench color theme. The matching palette lives in
 * `theme/themes.ts`; the active id is applied to `<html data-theme>` so the
 * semantic-token conditions in `theme/system.ts` resolve to the right colors.
 */
export type WorkbenchThemeId = 'dark' | 'light' | 'forest' | 'mono' | 'ultradark';

/** User-tunable appearance + behavior preferences surfaced in the Settings modal. */
export interface WorkbenchPreferences {
  themeId: WorkbenchThemeId;
  reduceMotion: boolean;
  showFocusRegionHighlight: boolean;
  confirmImageDeletion: boolean;
}

export interface AccountState {
  activeLayoutPresetId: LayoutPresetId;
  preferences: WorkbenchPreferences;
}

export interface WorkbenchPersistenceSnapshot {
  version: 1;
  savedAt: string;
  state: WorkbenchState;
}
