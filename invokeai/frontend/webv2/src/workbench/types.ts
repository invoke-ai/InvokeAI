import type { ComponentType, ExoticComponent, JSXElementConstructor, SVGProps } from 'react';

import type { ProjectGraphState } from './workflows/types';

export type LayoutPresetId = 'canvas-default' | 'gallery' | 'workflow' | 'linear';

export type CenterViewId = 'canvas' | 'gallery' | 'preview' | 'workflow';

export type GraphId = string;

export type FirstPartyWidgetTypeId =
  | 'autosave-status'
  | 'canvas'
  | 'diagnostics'
  | 'gallery'
  | 'generate'
  | 'history-controls'
  | 'layers'
  | 'notifications'
  | 'preview'
  | 'project'
  | 'queue'
  | 'server-status'
  | 'users'
  | 'version-status'
  | 'workflow';

export type WidgetTypeId = FirstPartyWidgetTypeId | (string & {});
export type WidgetInstanceId = string;
export type WidgetId = WidgetTypeId;

export type WorkbenchRegion = 'left' | 'right' | 'center' | 'bottom' | 'dialog' | 'popover';

export type QueueItemStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export type ProjectEventType =
  | 'project-created'
  | 'layout-updated'
  | 'invocation-updated'
  | 'queue-submitted'
  | 'canvas-layer-accepted'
  | 'graph-replaced'
  | 'graph-snapshot-saved';

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
  id: WidgetTypeId;
  label: string;
  version: 1;
  values: Record<string, unknown>;
  graphId?: GraphId;
}

export type WidgetStateMap = Record<string, WidgetStateContract>;

export interface WidgetInstanceContract {
  id: WidgetInstanceId;
  typeId: WidgetTypeId;
  title?: string;
  state: WidgetStateContract;
  createdAt: string;
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
  instance: WidgetInstanceContract;
  runtime: WidgetRuntimeApi;
  presentation?: 'compact' | 'expanded' | 'tooltip';
}

export type WidgetView = ComponentType<WidgetViewProps>;

export type WidgetHeaderActions = ComponentType<WidgetViewProps>;
export type WidgetHeaderMenu = ComponentType<WidgetViewProps>;
export type WidgetFooter = ComponentType<WidgetViewProps>;

export interface WidgetLabelProps {
  region: WorkbenchRegion;
  presentation?: 'compact' | 'expanded' | 'tooltip';
}

export type WidgetLabel = string | ComponentType<WidgetLabelProps>;

export type WidgetIconComponent =
  | JSXElementConstructor<SVGProps<SVGSVGElement>>
  | ExoticComponent<SVGProps<SVGSVGElement>>;

export interface WidgetStateRegistration<State extends Record<string, unknown> = Record<string, unknown>> {
  version: 1;
  createInitial: () => State;
  migrate?: (state: unknown, fromVersion: number) => State;
  persistence?: 'project' | 'workspace' | 'session' | 'none';
}

export interface WidgetRuntimeApi<State extends Record<string, unknown> = Record<string, unknown>> {
  instanceId: WidgetInstanceId;
  typeId: WidgetTypeId;
  region: WorkbenchRegion;
  state: State;
  patchState: (values: Partial<State>) => void;
  setState: (values: State) => void;
  commands: WidgetCommandApi;
  hotkeys: WidgetHotkeyApi;
  menus: WidgetMenuApi;
  palette: WidgetCommandPaletteApi;
  search: WidgetSearchApi;
  toolbars: WidgetToolbarApi;
  workbench: WidgetWorkbenchApi;
}

export interface WidgetCommandApi {
  execute: (commandId: string, ...args: unknown[]) => Promise<unknown>;
  register: (command: WidgetCommandContribution) => () => void;
}

export interface WidgetCommandContribution {
  id: string;
  title: string;
  handler: (...args: unknown[]) => unknown | Promise<unknown>;
}

export interface WidgetHotkeyApi {
  register: (hotkey: WidgetHotkeyContribution) => () => void;
}

export interface WidgetHotkeyContribution {
  commandId: string;
  keybinding: string;
  when?: string;
}

export interface WidgetMenuApi {
  register: (menu: WidgetMenuContribution) => () => void;
}

export interface WidgetMenuContribution {
  id: string;
  items: Array<{ commandId: string; group?: string }>;
}

export interface WidgetCommandPaletteApi {
  register: (entry: WidgetCommandPaletteContribution) => () => void;
}

export interface WidgetCommandPaletteContribution {
  commandId: string;
  title: string;
  keywords?: string[];
}

export interface WidgetSearchApi {
  registerProvider: (provider: WidgetSearchProvider) => () => void;
}

export interface WidgetSearchProvider {
  id: string;
  label: string;
  search: (query: string) => Promise<WidgetSearchResult[]> | WidgetSearchResult[];
}

export interface WidgetSearchResult {
  id: string;
  title: string;
  subtitle?: string;
  commandId?: string;
}

export interface WidgetToolbarApi {
  register: (toolbar: WidgetToolbarContribution) => () => void;
}

export interface WidgetToolbarContribution {
  id: string;
  location: 'center.tabs.trailing' | 'status.left' | 'status.right';
  items: Array<{ commandId: string; icon?: WidgetIconComponent; label?: string }>;
}

export interface OpenWorkbenchWidgetOptions {
  createNew?: boolean;
  preferredRegions?: ReadonlyArray<WidgetRegion>;
  requireCenterView?: boolean;
}

export type WidgetWorkbenchApiResult =
  | { ok: true; region?: WidgetRegion }
  | { ok: false; reason: 'unavailable' | 'unsupported' | 'not-found' };

export interface WidgetWorkbenchApi {
  openWidget: (typeId: WidgetTypeId, options?: OpenWorkbenchWidgetOptions) => WidgetWorkbenchApiResult;
  revealWidgetInstance: (instanceId: WidgetInstanceId) => WidgetWorkbenchApiResult;
  closeWidgetInstance: (instanceId: WidgetInstanceId) => WidgetWorkbenchApiResult;
}

/** Sections of the workbench settings dialog, addressable via `openWorkbenchSettings`. */
export type SettingsSectionId = 'appearance' | 'behavior' | 'project' | 'workflow' | 'developer' | 'workspace';

export interface WidgetManifest {
  /** Widget runtime API contract version. Defaults to 1 during registry normalization. */
  apiVersion?: 1;
  id: WidgetTypeId;
  label: WidgetLabel;
  labelText: string;
  version: 1;
  allowedRegions: WidgetRegion[];
  allowMultiple: boolean;
  icon: WidgetIconComponent;
  bottomPanel?: 'expandable' | 'tooltip';
  centerPlacement?: 'toolbar' | 'view';
  /** Only offered while an admin is signed in to a multi-user backend. */
  requiresAdmin?: boolean;
  chrome?: {
    header?: 'hidden' | 'visible';
  };
  view?: WidgetView;
  headerActions?: WidgetHeaderActions;
  /**
   * Extra entries for the widget's shared header actions menu. Rendered inside
   * the same menu as the universal graph actions, so widgets extend the frame
   * menu instead of stacking their own. Render `Menu.Item`/`Menu.ItemGroup`
   * children only; own any dialogs from `headerActions` (always mounted).
   */
  headerMenu?: WidgetHeaderMenu;
  /** When set, the frame header shows a gear that opens this settings dialog section. */
  settingsSection?: SettingsSectionId;
  footer?: WidgetFooter;
  state?: WidgetStateRegistration;
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

export interface NormalizedWidgetManifest extends Omit<WidgetManifest, 'apiVersion' | 'state'> {
  apiVersion: 1;
  state: WidgetStateRegistration;
}

export interface RegisteredWidget {
  manifest: NormalizedWidgetManifest;
  status: 'enabled' | 'disabled' | 'hidden';
  failure?: WidgetFailure;
}

export interface WidgetFailure {
  widgetId: WidgetTypeId;
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
  /** The top validation issue, shown on the fixed Invoke control's secondary line. */
  validationMessage?: string;
  /** Every reason the route cannot run right now (legacy `reasonsWhyCannotEnqueue` equivalent). */
  validationReasons: string[];
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
  activeInstanceId: WidgetInstanceId;
  instanceIds: WidgetInstanceId[];
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
  settings: ProjectSettings;
  layout: ProjectLayoutState;
  invocation: InvocationControllerState;
  /** The one active project graph: an editable workflow document, compiled to a `GraphContract` at invoke time. */
  projectGraph: ProjectGraphState;
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
  widgetGraphs: Partial<Record<WidgetTypeId, GraphContract>>;
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
  /** Widget to focus in the left rail when the preset is applied (e.g. the Linear UI for workflow presets). */
  leftRegionWidgetId?: WidgetId;
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

/**
 * One entry of the project's graph history. Queue submissions record the
 * compiled `graph`; workflow snapshots (manual save, pre-replacement) record
 * the editable `document`, which is what makes them restorable.
 */
export interface GraphHistorySnapshot {
  id: string;
  createdAt: string;
  label: string;
  graph?: GraphContract;
  document?: ProjectGraphState;
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
  projectGraph: ProjectGraphState;
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
  widgetGraphs: Partial<Record<WidgetTypeId, GraphContract>>;
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
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>;
  widgetStates: WidgetStateMap;
  canvas: CanvasStateContract;
  submittedAt: string;
}

export interface QueueItem {
  id: string;
  status: QueueItemStatus;
  cancellable: boolean;
  snapshot: QueueSubmissionSnapshot;
  backendItemIds?: number[];
  completedBackendItemIds?: number[];
  cancelledBackendItemIds?: number[];
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
export type WorkbenchThemeId = 'classic' | 'light' | 'forest' | 'mono' | 'ultradark';

export type WorkbenchLanguage =
  | 'ar'
  | 'az'
  | 'de'
  | 'en'
  | 'es'
  | 'fi'
  | 'fr'
  | 'he'
  | 'hu'
  | 'it'
  | 'ja'
  | 'ko'
  | 'nl'
  | 'pl'
  | 'pt'
  | 'pt-BR'
  | 'ru'
  | 'sv'
  | 'tr'
  | 'ua'
  | 'vi'
  | 'zh-CN'
  | 'zh-Hant';

export type DeveloperLogLevel = 'trace' | 'debug' | 'info' | 'warn' | 'error' | 'fatal';

export type DeveloperLogNamespace =
  | 'canvas'
  | 'canvas-workflow-integration'
  | 'config'
  | 'dnd'
  | 'events'
  | 'gallery'
  | 'generation'
  | 'metadata'
  | 'models'
  | 'system'
  | 'queue'
  | 'workflows';

export interface ProjectSettings {
  useCpuNoise: boolean;
  showProgressDetails: boolean;
  antialiasProgressImages: boolean;
  showProgressImagesInViewer: boolean;
  preferNumericAttentionStyle: boolean;
}

/** User-tunable appearance + behavior preferences surfaced in the Settings modal. */
export interface WorkbenchPreferences {
  themeId: WorkbenchThemeId;
  reduceMotion: boolean;
  showFocusRegionHighlight: boolean;
  confirmImageDeletion: boolean;
  language: WorkbenchLanguage;
  enableInformationalPopovers: boolean;
  enableModelDescriptions: boolean;
  developerLogEnabled: boolean;
  developerLogLevel: DeveloperLogLevel;
  developerLogNamespaces: DeveloperLogNamespace[];
  /** Always snap workflow nodes to the grid (Ctrl snaps temporarily when off). */
  workflowSnapToGrid: boolean;
  /** Show the minimap in the workflow editor. */
  workflowShowMinimap: boolean;
  /** Reject workflow connections with incompatible field types. */
  workflowValidateConnections: boolean;
  /** Connection line rendering in the workflow editor. */
  workflowEdgeStyle: 'curved' | 'square';
}

export interface AccountState {
  activeLayoutPresetId: LayoutPresetId;
}

export interface WorkbenchPersistenceSnapshot {
  version: 1;
  savedAt: string;
  state: WorkbenchState;
}
