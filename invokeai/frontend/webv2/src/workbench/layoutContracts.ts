import type { WidgetInstanceId, WidgetTypeId } from './widgetContracts';

export type BuiltInLayoutPresetId = 'canvas-default' | 'gallery' | 'workflow' | 'canvas';

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

export interface LayoutPreset {
  id: LayoutPresetId;
  label: string;
  isBuiltIn?: boolean;
  snapshot: LayoutPresetSnapshot;
}
