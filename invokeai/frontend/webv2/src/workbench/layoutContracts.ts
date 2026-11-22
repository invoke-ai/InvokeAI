import type { InvocationSourceId, ResultDestination } from './invocationContracts';
import type { WidgetInstanceId, WidgetTypeId } from './widgetContracts';

/** Presets name arrangements, not invocation sources; graphWidgets owns source resolution. */
export type BuiltInLayoutPresetId = 'compose' | 'edit' | 'automate' | 'video';

export type LayoutPresetId = BuiltInLayoutPresetId | (string & {});

/** The stored snapshot's center view field: older builds validate it, so writes keep it naming the stage's type. */
export type CenterViewId = 'canvas' | 'gallery' | 'preview' | 'workflow';

export interface PanelState {
  isLeftOpen: boolean;
  isRightOpen: boolean;
  isBottomOpen: boolean;
}

export type WidgetRegion = 'left' | 'right' | 'bottom' | 'center';

export const WIDGET_REGIONS: readonly WidgetRegion[] = ['left', 'right', 'bottom', 'center'];

export const isWidgetRegion = (value: unknown): value is WidgetRegion =>
  typeof value === 'string' && (WIDGET_REGIONS as readonly string[]).includes(value);

export type FloatingWidgetMode = 'windowed' | 'maximized' | 'shaded';

export interface FloatingWidgetState {
  x: number;
  y: number;
  widthPx: number;
  heightPx: number;
  mode: FloatingWidgetMode;
  returnRegion: WidgetRegion;
  /**
   * Position in the return region's complete order: its docked members plus one marker per floating window. Placement
   * normalization makes it canonical; only data stored by an older build lacks it or disagrees.
   */
  returnIndex?: number;
  /** Z-order within the floating layer; higher renders on top. */
  stackOrder: number;
}

/** Where a window sat and how large it was, kept after it docks or closes so floating it again reopens it there. */
export type FloatingWidgetGeometry = Pick<FloatingWidgetState, 'heightPx' | 'widthPx' | 'x' | 'y'>;

export interface WidgetRegionState {
  activeInstanceId: WidgetInstanceId;
  instanceIds: WidgetInstanceId[];
  /** Instances rendered in the strip's trailing cluster (bottom region only).
   * Ids not currently placed are inert — a widget re-enabled later keeps its side. */
  alignEndInstanceIds?: WidgetInstanceId[];
  isCollapsed: boolean;
  sizePx: number;
}

export interface LayoutPresetSnapshotLayout {
  presetId: LayoutPresetId;
  /** Never read: the center's home front is the stage. Written consistent with it, as older builds require. */
  centerViewId: CenterViewId;
  panels: PanelState;
}

export interface LayoutPresetWidgetInstanceSnapshot {
  id: WidgetInstanceId;
  typeId: WidgetTypeId;
  title?: string;
}

export interface LayoutPresetSnapshot {
  layout: LayoutPresetSnapshotLayout;
  widgetInstances: Record<WidgetInstanceId, LayoutPresetWidgetInstanceSnapshot>;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
  /** Capture windows with widgetRegions so floated instances survive presets; absent means none. */
  floatingWidgets?: Record<WidgetInstanceId, FloatingWidgetState>;
}

/**
 * The route a preset establishes when it is activated. Locks belong to the live project controller and are
 * intentionally never persisted on a preset.
 */
export interface LayoutPresetRoute {
  sourceId: InvocationSourceId;
  destination: ResultDestination;
}

export interface LayoutPreset {
  id: LayoutPresetId;
  label: string;
  /**
   * The i18n key a built-in preset is shown by while the account has not renamed it; `label` is then its English
   * default. Never on a custom preset, and never persisted.
   */
  labelKey?: string;
  isBuiltIn?: boolean;
  /** Missing only on legacy or source-less custom presets. */
  defaultRoute?: LayoutPresetRoute;
  /** Persist an icon id, resolved through the picker registry with an unknown-id fallback. */
  iconId?: string;
  snapshot: LayoutPresetSnapshot;
}

/** Account overrides: the structures of code-owned built-ins as this account customized them. */
export type LayoutPresetOverrides = Partial<Record<BuiltInLayoutPresetId, LayoutPresetSnapshot>>;

/** Per-account edits to the default routes shipped with built-in presets. */
export type LayoutPresetRouteOverrides = Partial<Record<BuiltInLayoutPresetId, LayoutPresetRoute>>;

/** Per-account edits to the identity shipped with a built-in preset. */
export interface LayoutPresetMetadataOverride {
  iconId?: string;
  label?: string;
}

export type LayoutPresetMetadataOverrides = Partial<Record<BuiltInLayoutPresetId, LayoutPresetMetadataOverride>>;
