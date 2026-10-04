/** Pure layout contracts for the Layers widget's pane blocks; the manifest seeds them, so no React here. */

export type LayerEditorPaneId = 'properties' | 'transform' | 'overview';

/** What every pane block persists: its preferred height and whether it is collapsed to its strip. */
export interface PaneBlockLayout {
  isCollapsed: boolean;
  sizePx: number;
}

export interface LayerEditorPaneLayout extends PaneBlockLayout {
  activePane: LayerEditorPaneId;
}

export const LAYER_EDITOR_PANE_MIN_SIZE_PX = 140;
export const LAYER_EDITOR_PANE_MAX_SIZE_PX = 560;

export const LAYER_EDITOR_PANE_DEFAULTS: LayerEditorPaneLayout = {
  activePane: 'properties',
  isCollapsed: false,
  sizePx: 300,
};

export const clampLayerEditorPaneSize = (sizePx: number): number =>
  Math.min(LAYER_EDITOR_PANE_MAX_SIZE_PX, Math.max(LAYER_EDITOR_PANE_MIN_SIZE_PX, Math.round(sizePx)));

const isPaneId = (value: unknown): value is LayerEditorPaneId =>
  value === 'properties' || value === 'transform' || value === 'overview';

/** The persisted widget values are untyped; anything malformed falls back per field. */
export const readLayerEditorPaneLayout = (values: Record<string, unknown>): LayerEditorPaneLayout => {
  const raw = values.editorPanes;
  const layout = typeof raw === 'object' && raw !== null ? (raw as Partial<LayerEditorPaneLayout>) : {};
  return {
    activePane: isPaneId(layout.activePane) ? layout.activePane : LAYER_EDITOR_PANE_DEFAULTS.activePane,
    isCollapsed: typeof layout.isCollapsed === 'boolean' ? layout.isCollapsed : LAYER_EDITOR_PANE_DEFAULTS.isCollapsed,
    sizePx:
      typeof layout.sizePx === 'number' && Number.isFinite(layout.sizePx)
        ? clampLayerEditorPaneSize(layout.sizePx)
        : LAYER_EDITOR_PANE_DEFAULTS.sizePx,
  };
};

export const areLayerEditorPaneLayoutsEqual = (a: LayerEditorPaneLayout, b: LayerEditorPaneLayout): boolean =>
  a.activePane === b.activePane && a.isCollapsed === b.isCollapsed && a.sizePx === b.sizePx;

export type LayerColorPaneId = 'color' | 'swatches';

export interface LayerColorPaneLayout extends PaneBlockLayout {
  activePane: LayerColorPaneId;
}

export const COLOR_PANE_MIN_SIZE_PX = 160;
export const COLOR_PANE_MAX_SIZE_PX = 560;

export const COLOR_PANE_DEFAULTS: LayerColorPaneLayout = {
  activePane: 'color',
  isCollapsed: false,
  sizePx: 236,
};

/** Defaults from the unreleased taller layouts; a stored exact match adopts the current default. */
const LEGACY_COLOR_PANE_DEFAULT_SIZES = new Set([300, 420]);

export const clampColorPaneSize = (sizePx: number): number =>
  Math.min(COLOR_PANE_MAX_SIZE_PX, Math.max(COLOR_PANE_MIN_SIZE_PX, Math.round(sizePx)));

const isColorPaneId = (value: unknown): value is LayerColorPaneId => value === 'color' || value === 'swatches';

export const readColorPaneLayout = (values: Record<string, unknown>): LayerColorPaneLayout => {
  const raw = values.colorPane;
  const layout = typeof raw === 'object' && raw !== null ? (raw as Partial<LayerColorPaneLayout>) : {};
  return {
    activePane: isColorPaneId(layout.activePane) ? layout.activePane : COLOR_PANE_DEFAULTS.activePane,
    isCollapsed: typeof layout.isCollapsed === 'boolean' ? layout.isCollapsed : COLOR_PANE_DEFAULTS.isCollapsed,
    sizePx:
      typeof layout.sizePx === 'number' &&
      Number.isFinite(layout.sizePx) &&
      !LEGACY_COLOR_PANE_DEFAULT_SIZES.has(layout.sizePx)
        ? clampColorPaneSize(layout.sizePx)
        : COLOR_PANE_DEFAULTS.sizePx,
  };
};

export const areColorPaneLayoutsEqual = (a: LayerColorPaneLayout, b: LayerColorPaneLayout): boolean =>
  a.activePane === b.activePane && a.isCollapsed === b.isCollapsed && a.sizePx === b.sizePx;

export type LayerTreeTabId = 'layers' | 'history';

export const LAYER_TREE_TAB_DEFAULT: LayerTreeTabId = 'layers';

export const readLayerTreeTab = (values: Record<string, unknown>): LayerTreeTabId =>
  values.treeTab === 'history' ? 'history' : LAYER_TREE_TAB_DEFAULT;
