import type { InvocationSourceId, ResultDestination } from './invocationContracts';
import type { WidgetInstanceId, WidgetTypeId } from './widgetContracts';

/** Presets name arrangements, not invocation sources; graphWidgets owns source resolution. */
export type BuiltInLayoutPresetId = 'compose' | 'edit' | 'automate' | 'video';

export type LayoutPresetId = BuiltInLayoutPresetId | (string & {});

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

/** Geometry + stacking for a widget instance detached into a floating window. */
export interface FloatingWidgetState {
  x: number;
  y: number;
  widthPx: number;
  heightPx: number;
  mode: FloatingWidgetMode;
  /** The dockable region this window returns to when docked. */
  returnRegion: WidgetRegion;
  /** Original rail index for docking; absent or invalid indices append. */
  returnIndex?: number;
  /** Z-order within the floating layer; higher renders on top. */
  stackOrder: number;
}

export interface WidgetRegionState {
  activeInstanceId: WidgetInstanceId;
  instanceIds: WidgetInstanceId[];
  /** Instances rendered in the strip's trailing cluster (bottom region only).
   * Ids not currently placed are inert — a widget re-enabled later keeps its side. */
  alignEndInstanceIds?: WidgetInstanceId[];
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
  /** Capture windows with widgetRegions so floated instances survive presets; absent means none. */
  floatingWidgets?: Record<WidgetInstanceId, FloatingWidgetState>;
}

/**
 * The route a preset establishes when it is activated. Locks belong to the
 * live project controller and are intentionally never persisted on a preset.
 */
export interface LayoutPresetRoute {
  sourceId: InvocationSourceId;
  destination: ResultDestination;
}

export interface LayoutPreset {
  id: LayoutPresetId;
  label: string;
  isBuiltIn?: boolean;
  /** Missing only on legacy or source-less custom presets. */
  defaultRoute?: LayoutPresetRoute;
  /** Persist an icon id, resolved through the picker registry with an unknown-id fallback. */
  iconId?: string;
  snapshot: LayoutPresetSnapshot;
}

/** Account overrides store edits to code-owned built-ins and drive both drift checks and revert. */
export type LayoutPresetOverrides = Partial<Record<BuiltInLayoutPresetId, LayoutPresetSnapshot>>;

/** Per-account edits to the default routes shipped with built-in presets. */
export type LayoutPresetRouteOverrides = Partial<Record<BuiltInLayoutPresetId, LayoutPresetRoute>>;

/** Per-account edits to the identity shipped with a built-in preset. */
export interface LayoutPresetMetadataOverride {
  iconId?: string;
  label?: string;
}

export type LayoutPresetMetadataOverrides = Partial<Record<BuiltInLayoutPresetId, LayoutPresetMetadataOverride>>;
