import type { ComponentType } from 'react';

export type LayoutPresetId = 'canvas-default' | 'gallery' | 'workflow' | 'linear' | 'model-manager';

export type CenterViewId = 'canvas' | 'gallery' | 'preview' | 'workflow';

export type GraphId = string;

export type WidgetId =
  | 'autosave-status'
  | 'canvas'
  | 'gallery'
  | 'generate'
  | 'history-controls'
  | 'layout-actions'
  | 'layers'
  | 'queue'
  | 'server-status'
  | 'version-status'
  | 'workflow';

export type WorkbenchRegion = 'left' | 'right' | 'center' | 'bottom' | 'dialog' | 'popover';

export type QueueItemStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export type ProjectEventType = 'project-created' | 'layout-updated' | 'invocation-updated' | 'queue-submitted';

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
  view?: WidgetView;
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

export interface CanvasStateContract {
  version: 1;
  layers: string[];
  stagingArea: {
    selectedLayerId?: string;
    pendingImageIds: string[];
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
  errorLog: string[];
  autosave: AutosaveState;
  account: AccountState;
  widgetFailures: WidgetFailure[];
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

export interface AccountState {
  activeLayoutPresetId: LayoutPresetId;
}

export interface WorkbenchPersistenceSnapshot {
  version: 1;
  savedAt: string;
  state: WorkbenchState;
}
