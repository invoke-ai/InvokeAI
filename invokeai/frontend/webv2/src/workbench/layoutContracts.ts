import type { InvocationSourceId, ResultDestination } from './invocationContracts';
import type { WidgetInstanceId, WidgetTypeId } from './widgetContracts';

/**
 * The three shipped layout presets. A preset names an *arrangement*, never a
 * widget — `edit` opens Canvas but is not "the Canvas view", and a preset may
 * have several graph widgets open at once. Source resolution is a separate
 * concern (see `graphWidgets.ts`).
 */
export type BuiltInLayoutPresetId = 'compose' | 'edit' | 'automate';

export type LayoutPresetId = BuiltInLayoutPresetId | (string & {});

export type CenterViewId = 'canvas' | 'gallery' | 'preview' | 'workflow';

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
  /**
   * Icon identifier, not a component: presets are persisted verbatim, so this
   * has to survive a round trip through storage. Resolved against the picker's
   * registry, which falls back for ids it no longer knows.
   */
  iconId?: string;
  snapshot: LayoutPresetSnapshot;
}

/**
 * Per-account edits to a preset's saved arrangement, keyed by preset id. Built-in
 * preset bodies are code, so "Save changes" cannot mutate them in place; the
 * override is what the drift comparison and `Revert to saved layout` read.
 */
export type LayoutPresetOverrides = Partial<Record<BuiltInLayoutPresetId, LayoutPresetSnapshot>>;

/** Per-account edits to the default routes shipped with built-in presets. */
export type LayoutPresetRouteOverrides = Partial<Record<BuiltInLayoutPresetId, LayoutPresetRoute>>;

/** Per-account edits to the identity shipped with a built-in preset. */
export interface LayoutPresetMetadataOverride {
  iconId?: string;
  label?: string;
}

export type LayoutPresetMetadataOverrides = Partial<Record<BuiltInLayoutPresetId, LayoutPresetMetadataOverride>>;
