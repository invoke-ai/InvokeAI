import type { TFunction } from 'i18next';
import type { ComponentType, ExoticComponent, JSXElementConstructor, SVGProps } from 'react';

// `generation/types.ts` imports value-less (`import type`) symbols back from this
// module; both sides are type-only so this doesn't create a runtime cycle.
import type { GenerateReferenceImage, GenerateWidgetValues } from './generation/types';
import type { WorkbenchLanguage } from './i18n/languages';
import type { ProjectGraphState } from './workflows/types';

export type { WorkbenchLanguage } from './i18n/languages';

export type BuiltInLayoutPresetId = 'canvas-default' | 'gallery' | 'workflow' | 'canvas';
export type LayoutPresetId = BuiltInLayoutPresetId | (string & {});

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

export interface WidgetInstanceRuntimeMeta {
  id: WidgetInstanceId;
  typeId: WidgetTypeId;
  title?: string;
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
  /** Chrome metadata only. Read reactive widget values through `runtime.state`. */
  instance: WidgetInstanceRuntimeMeta;
  runtime: WidgetRuntimeApi;
  presentation?: 'compact' | 'expanded' | 'tooltip';
}

export type WidgetView = ComponentType<WidgetViewProps>;

export type WidgetHeaderActions = ComponentType<WidgetViewProps>;
export type WidgetHeaderMenu = ComponentType<WidgetViewProps>;
export type WidgetFooter = ComponentType<WidgetViewProps>;
export type WidgetHost = ComponentType;

export interface WidgetLabelProps {
  region: WorkbenchRegion;
  presentation?: 'compact' | 'expanded' | 'tooltip';
}

export type WidgetLabel = string | ((t: TFunction) => string);
export type WidgetHeaderLabel = ComponentType<WidgetLabelProps>;

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
  state: WidgetRuntimeStateApi<State>;
  diagnostics: WidgetDiagnosticsApi;
  commands: WidgetCommandApi;
  hotkeys: WidgetHotkeyApi;
  menus: WidgetMenuApi;
  palette: WidgetCommandPaletteApi;
  search: WidgetSearchApi;
  toolbars: WidgetToolbarApi;
  workbench: WidgetWorkbenchApi;
}

export interface WidgetDiagnosticsApi {
  logger: (namespace: DeveloperLogNamespace) => {
    debug: (messageOrContext: string | Record<string, unknown>, message?: string) => void;
    error: (messageOrContext: string | Record<string, unknown>, message?: string) => void;
    fatal: (messageOrContext: string | Record<string, unknown>, message?: string) => void;
    info: (messageOrContext: string | Record<string, unknown>, message?: string) => void;
    trace: (messageOrContext: string | Record<string, unknown>, message?: string) => void;
    warn: (messageOrContext: string | Record<string, unknown>, message?: string) => void;
  };
}

export interface WidgetRuntimeStateApi<State extends Record<string, unknown> = Record<string, unknown>> {
  getSnapshot: () => Readonly<State>;
  useSelector: <Selected>(
    selector: (state: Readonly<State>) => Selected,
    isEqual?: (left: Selected, right: Selected) => boolean
  ) => Selected;
  patch: (values: Partial<State>) => void;
  set: (values: State) => void;
}

export interface WidgetContributionSource {
  instanceId: WidgetInstanceId;
  projectId: string;
  region: WorkbenchRegion;
  typeId: WidgetTypeId;
}

export interface WidgetCommandApi {
  execute: (commandId: string, ...args: unknown[]) => Promise<unknown>;
  executeForSource: (
    commandId: string,
    source: WidgetContributionSource | null,
    ...args: unknown[]
  ) => Promise<unknown>;
  register: (command: WidgetCommandContribution) => () => void;
}

export interface WidgetCommandContribution {
  id: string;
  title: string;
  handler: (...args: unknown[]) => unknown | Promise<unknown>;
  source?: WidgetContributionSource;
}

export interface WidgetHotkeyApi {
  register: (hotkey: WidgetHotkeyContribution) => () => void;
}

export interface WidgetHotkeyContribution {
  id: string;
  commandId: string;
  defaultKeys: string[];
  title: string;
  description?: string;
  scope?: 'focused-region' | 'global' | 'instance' | 'widget';
  source?: WidgetContributionSource;
  preventDefault?: boolean;
  allowInEditable?: boolean;
}

export interface WidgetMenuApi {
  register: (menu: WidgetMenuContribution) => () => void;
}

export interface WidgetMenuContribution {
  id: string;
  items: Array<{ commandId: string; group?: string }>;
  source?: WidgetContributionSource;
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
  items: Array<{
    commandId: string;
    icon?: WidgetIconComponent;
    label?: string;
  }>;
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
export type SettingsSectionId =
  | 'appearance'
  | 'behavior'
  | 'hotkeys'
  | 'project'
  | 'queue'
  | 'workflow'
  | 'developer'
  | 'workspace';

export interface WidgetManifest {
  /** Widget runtime API contract version. Defaults to 1 during registry normalization. */
  apiVersion?: 1;
  id: WidgetTypeId;
  label: WidgetLabel;
  headerLabel?: WidgetHeaderLabel;
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
  /**
   * Stable singleton UI owned by the widget but not by any widget instance.
   * Use for dialogs, file inputs, and other portals driven by widget-level
   * stores. Mounted once under WorkbenchProvider, independent of widget chrome.
   */
  host?: WidgetHost;
  headerActions?: WidgetHeaderActions;
  /**
   * Extra entries for the widget's shared header actions menu. Rendered inside
   * the same menu as the universal graph actions, so widgets extend the frame
   * menu instead of stacking their own. Render `Menu.Item`/`Menu.ItemGroup`
   * children only; use `host` for stable dialogs or file inputs.
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
  sourceBackendItemId?: number;
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

/** @deprecated legacy v1 shape, superseded by {@link CanvasStagingAreaContractV2}. Extracted so v1 and v2 canvas state can share its structure. */
export interface CanvasStagingAreaContract {
  sourceQueueItemId?: string;
  selectedLayerId?: string;
  pendingImageIds: string[];
  pendingImages: CanvasStagingCandidateContract[];
  selectedImageIndex: number;
  isVisible: boolean;
  areThumbnailsVisible: boolean;
}

/** @deprecated legacy v1 shape, superseded by {@link CanvasStateContractV2}. Kept for migration input typing and the still-v1-shaped staging area. */
export interface CanvasStateContract {
  version: 1;
  document: CanvasDocumentContract;
  stagingArea: CanvasStagingAreaContract;
}

// ---------------------------------------------------------------------------
// Canvas v2 document contracts
//
// The canvas document is being rebuilt as a full photo editor: a layer stack
// supporting raster, control, regional-guidance, and inpaint-mask layers, each
// with its own transform/opacity/blend-mode. `CanvasStateContract` (v1) only
// ever produced single-image raster layers positioned by `placement`; v2
// layers are positioned by `transform` and reference bitmaps by `imageName`
// (via `CanvasImageRef`) rather than by resolved URLs, since URLs are
// ephemeral/backend-derived while the document is persisted. The staging
// area (pending queue results awaiting placement onto the canvas) keeps its
// v1 shape unchanged — only the accepted document evolves.
// ---------------------------------------------------------------------------

export type CanvasBlendMode =
  | 'normal'
  | 'multiply'
  | 'screen'
  | 'overlay'
  | 'darken'
  | 'lighten'
  | 'color-dodge'
  | 'color-burn'
  | 'hard-light'
  | 'soft-light'
  | 'difference'
  | 'exclusion'
  | 'hue'
  | 'saturation'
  | 'color'
  | 'luminosity';

/** A reference to a persisted image asset by name, not by resolved URL. */
export interface CanvasImageRef {
  imageName: string;
  width: number;
  height: number;
  contentHash?: string;
}

export type CanvasLayerSourceContract =
  | {
      type: 'paint';
      bitmap: CanvasImageRef | null;
      /**
       * The layer-local origin of `bitmap`'s top-left pixel. Paint layers are
       * content-sized: the persisted bitmap covers only the painted region, and
       * this offset records where that region sits in the layer's local space
       * (it can be negative). Absent (or `{ x: 0, y: 0 }`) for legacy documents
       * whose paint bitmaps were document-sized at the origin — they load
       * identically.
       */
      offset?: { x: number; y: number };
    }
  | { type: 'image'; image: CanvasImageRef }
  | {
      type: 'text';
      content: string;
      fontFamily: string;
      fontSize: number;
      fontWeight: number;
      lineHeight: number;
      align: 'left' | 'center' | 'right';
      color: string;
    }
  | {
      type: 'shape';
      kind: 'rect' | 'ellipse' | 'polygon';
      points?: { x: number; y: number }[];
      width: number;
      height: number;
      fill: string | null;
      stroke: string | null;
      strokeWidth: number;
    }
  | {
      type: 'gradient';
      kind: 'linear' | 'radial';
      angle: number;
      stops: { offset: number; color: string }[];
      /**
       * The gradient's explicit content extent (layer-local pixels). Gradient
       * layers are content-sized like every other layer: the extent is set at
       * creation (bbox-sized) and preserved across angle edits. Absent for legacy
       * documents whose gradients were document-sized by construction — they
       * default to the document dimensions on load (see `getSourceContentRect`).
       */
      width?: number;
      height?: number;
    };

export interface CanvasLayerBaseContract {
  id: string;
  name: string;
  isEnabled: boolean;
  isLocked: boolean;
  opacity: number;
  blendMode: CanvasBlendMode;
  transform: { x: number; y: number; scaleX: number; scaleY: number; rotation: number };
}

export interface CanvasAdjustmentsContract {
  brightness: number;
  contrast: number;
  saturation: number;
  curves?: { r: [number, number][]; g: [number, number][]; b: [number, number][] };
}

export interface CanvasControlAdapterContract {
  kind: 'controlnet' | 't2i_adapter' | 'control_lora' | 'z_image_control';
  model: string | null;
  weight: number;
  beginEndStepPct: [number, number];
  controlMode: 'balanced' | 'more_prompt' | 'more_control' | 'unbalanced' | null;
}

export interface CanvasMaskFillContract {
  style: 'solid' | 'grid' | 'crosshatch' | 'diagonal' | 'horizontal' | 'vertical';
  color: string;
}

export interface CanvasMaskContract {
  bitmap: CanvasImageRef | null;
  fill: CanvasMaskFillContract;
  /**
   * The layer-local origin of `bitmap`'s top-left pixel. Mask layers are
   * content-sized exactly like paint layers: the persisted mask bitmap covers
   * only the painted region, and this offset records where that region sits in
   * the layer's local space (it can be negative). Absent (or `{ x: 0, y: 0 }`)
   * for legacy documents whose mask bitmaps were document-sized at the origin —
   * they load identically. Mirrors {@link CanvasLayerSourceContract} `paint`.
   */
  offset?: { x: number; y: number };
}

export interface CanvasRasterLayerContractV2 extends CanvasLayerBaseContract {
  type: 'raster';
  source: CanvasLayerSourceContract;
  adjustments?: CanvasAdjustmentsContract;
  isTransparencyLocked?: boolean;
  filter?: { type: string; settings: Record<string, unknown> };
}

export interface CanvasControlLayerContract extends CanvasLayerBaseContract {
  type: 'control';
  source: CanvasLayerSourceContract;
  adapter: CanvasControlAdapterContract;
  withTransparencyEffect: boolean;
  filter?: { type: string; settings: Record<string, unknown> };
}

export interface CanvasRegionalGuidanceLayerContract extends CanvasLayerBaseContract {
  type: 'regional_guidance';
  mask: CanvasMaskContract;
  positivePrompt: string | null;
  negativePrompt: string | null;
  autoNegative: boolean;
  referenceImages: GenerateReferenceImage[];
}

export interface CanvasInpaintMaskLayerContract extends CanvasLayerBaseContract {
  type: 'inpaint_mask';
  mask: CanvasMaskContract;
  noiseLevel?: number;
  denoiseLimit?: number;
}

export type CanvasLayerContract =
  | CanvasRasterLayerContractV2
  | CanvasControlLayerContract
  | CanvasRegionalGuidanceLayerContract
  | CanvasInpaintMaskLayerContract;

export interface CanvasDocumentContractV2 {
  version: 2;
  width: number;
  height: number;
  background: 'transparent' | { color: string };
  /** Index 0 is the top-most layer. */
  layers: CanvasLayerContract[];
  bbox: { x: number; y: number; width: number; height: number };
  selectedLayerId: string | null;
}

export interface CanvasSnapshotContract {
  id: string;
  name: string;
  createdAt: string;
  document: CanvasDocumentContractV2;
}

export interface CanvasStagingAreaContractV2 extends CanvasStagingAreaContract {
  autoSwitchMode: 'off' | 'latest' | 'progress';
}

export interface CanvasStateContractV2 {
  version: 2;
  document: CanvasDocumentContractV2;
  /**
   * Monotonic counter bumped whenever the document is swapped wholesale
   * (snapshot restore, `replaceCanvasDocument`) rather than incrementally
   * edited. The document mirror treats any change to this value as a full
   * document replacement (clearing engine pixel history), even when the new
   * document keeps the same dimensions and reuses layer ids — the case a
   * reference/dimension diff alone cannot distinguish from an ordinary edit.
   */
  documentRevision: number;
  snapshots: CanvasSnapshotContract[];
  stagingArea: CanvasStagingAreaContractV2;
}

export type InvocationSourceId = 'generate' | 'workflow' | 'upscale' | 'canvas';

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

export interface LayoutPresetWidgetInstanceSnapshot {
  id: WidgetInstanceId;
  typeId: WidgetTypeId;
  title?: string;
}

export interface LayoutPresetSnapshot {
  layout: ProjectLayoutState;
  widgetInstances: Record<WidgetInstanceId, LayoutPresetWidgetInstanceSnapshot>;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
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
  canvas: CanvasStateContractV2;
  graphHistory: GraphHistorySnapshot[];
  promptHistory: PromptHistoryItem[];
  undoRedo: UndoRedoHistory;
  queue: QueueState;
  events: ProjectEvent[];
}

export interface LayoutPreset {
  id: LayoutPresetId;
  label: string;
  isBuiltIn?: boolean;
  snapshot: LayoutPresetSnapshot;
}

export interface WorkbenchState {
  projects: Project[];
  activeProjectId: string;
  backendConnection: BackendConnectionState;
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

export interface PromptHistoryItem {
  positivePrompt: string;
  negativePrompt: string | null;
}

export interface UndoRedoEntry {
  id: string;
  createdAt: string;
  label: string;
  project: ProjectUndoSnapshot;
}

/**
 * Project-level undo snapshot. Deliberately excludes `canvas`: the canvas
 * rendering engine owns its own pixel-patch history, so project undo/redo
 * passes the live `project.canvas` through untouched (see `restoreUndoSnapshot`).
 */
export interface ProjectUndoSnapshot {
  layout: ProjectLayoutState;
  invocation: InvocationControllerState;
  projectGraph: ProjectGraphState;
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
  widgetGraphs: Partial<Record<WidgetTypeId, GraphContract>>;
}

export interface UndoRedoHistory {
  past: UndoRedoEntry[];
  future: UndoRedoEntry[];
}

export interface QueueSubmissionSnapshot {
  sourceId: InvocationSourceId;
  destination: ResultDestination;
  graph: GraphContract;
  generate?: {
    values: GenerateWidgetValues;
    negativePromptNodeId: string;
    positivePromptNodeId: string;
    seedNodeId: string;
  };
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>;
  widgetStates: WidgetStateMap;
  canvas: CanvasStateContractV2;
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
  showPromptSyntaxHighlighting: boolean;
}

/** User-tunable appearance + behavior preferences surfaced in the Settings modal. */
export interface WorkbenchPreferences {
  themeId: WorkbenchThemeId;
  reduceMotion: boolean;
  showFocusRegionHighlight: boolean;
  confirmImageDeletion: boolean;
  queueJobsScope: 'active-project' | 'all';
  language: WorkbenchLanguage;
  enableInformationalPopovers: boolean;
  enableModelDescriptions: boolean;
  developerLogEnabled: boolean;
  developerLogLevel: DeveloperLogLevel;
  developerLogNamespaces: DeveloperLogNamespace[];
  developerPerformanceTimingsEnabled: boolean;
  /** Always snap workflow nodes to the grid (Ctrl snaps temporarily when off). */
  workflowSnapToGrid: boolean;
  /** Show the minimap in the workflow editor. */
  workflowShowMinimap: boolean;
  /** Reject workflow connections with incompatible field types. */
  workflowValidateConnections: boolean;
  /** Connection line rendering in the workflow editor. */
  workflowEdgeStyle: 'curved' | 'square';
  /** Account-bound overrides keyed by hotkey id (`app.invoke`, `gallery.galleryNavLeft`, etc.). */
  customHotkeys: Record<string, string[]>;
}

export interface AccountState {
  activeLayoutPresetId: LayoutPresetId;
  customLayoutPresets?: LayoutPreset[];
}

export interface WorkbenchPersistenceSnapshot {
  version: 1;
  savedAt: string;
  state: WorkbenchState;
}
