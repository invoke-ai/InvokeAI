/**
 * Canvas view settings — the sectioned, legacy-style popover of per-project
 * canvas preferences (Behavior / Display / Grid), plus the Shift-revealed Debug
 * actions rendered separately in the header.
 *
 * Each setting is a boolean persisted in the canvas widget's own state values
 * (`widgetInstances['canvas'].state.values[key]`) — the same plumbing as the
 * denoising strength (`invoke/canvasStrength.ts`), so it survives reloads and
 * rides along in queue snapshots. Persistence is per-user (per-project), NEVER in
 * the canvas undo history.
 *
 * A setting either drives an engine boolean store (`store` set — React resolves
 * the persisted value and feeds it down; see the settings-feed effect in
 * `CanvasWidgetView`, the engine reads only its stores and never React) OR is
 * consumed directly React-side (`store` absent — e.g. `showProgressOnCanvas`,
 * which gates the progress-preview feed in the widget shell). The feed is strictly
 * one-directional: settings → engine, never the reverse.
 *
 * The list is data-driven: adding a boolean setting is one entry here plus its
 * i18n label and (when engine-backed) the store it drives. Pure data + readers;
 * no React, no engine imports — unit-testable in node.
 */

/** Which engine boolean store a setting feeds (settings whose `store` is absent are React-consumed). */
export type CanvasSettingStore =
  | 'checkerboard'
  | 'showGrid'
  | 'invertBrushSizeScroll'
  | 'showBbox'
  | 'bboxOverlay'
  | 'ruleOfThirds'
  | 'snapToGrid';

/** The popover section a setting is grouped under. */
export type CanvasSettingSection = 'behavior' | 'display' | 'grid';

/** Persisted keys inside the canvas widget's `state.values`. */
export const CANVAS_CHECKERBOARD_KEY = 'showCheckerboard';
export const CANVAS_SHOW_GRID_KEY = 'showGrid';
export const CANVAS_INVERT_BRUSH_SCROLL_KEY = 'invertBrushSizeScroll';
export const CANVAS_SHOW_BBOX_KEY = 'showBbox';
export const CANVAS_BBOX_OVERLAY_KEY = 'bboxOverlay';
export const CANVAS_RULE_OF_THIRDS_KEY = 'ruleOfThirds';
export const CANVAS_SNAP_TO_GRID_KEY = 'snapToGrid';
export const CANVAS_SHOW_PROGRESS_KEY = 'showProgressOnCanvas';

/** A single boolean canvas setting: its persisted key, default, label, section, and (optional) engine store. */
export interface CanvasBooleanSetting {
  /** The persisted key inside the canvas widget's `state.values`. */
  key: string;
  /** The engine store this value drives; absent when the setting is consumed React-side. */
  store?: CanvasSettingStore;
  /** Default value applied when the key is unset. */
  defaultValue: boolean;
  /** i18n key for the popover label. */
  labelKey: string;
  /** Which popover section the setting belongs to. */
  section: CanvasSettingSection;
}

/**
 * The canvas settings, in popover order (grouped by section). Extend by adding an
 * entry (plus its label and, when engine-backed, a store); the popover,
 * persistence, and engine feed all derive from this list.
 */
export const CANVAS_SETTINGS: readonly CanvasBooleanSetting[] = [
  // ── Behavior ──────────────────────────────────────────────────────────────
  {
    defaultValue: false,
    key: CANVAS_INVERT_BRUSH_SCROLL_KEY,
    labelKey: 'widgets.canvas.settings.invertBrushScroll',
    section: 'behavior',
    store: 'invertBrushSizeScroll',
  },
  // ── Display ───────────────────────────────────────────────────────────────
  {
    defaultValue: true,
    key: CANVAS_SHOW_PROGRESS_KEY,
    labelKey: 'widgets.canvas.settings.showProgressOnCanvas',
    section: 'display',
    // No engine store: gated React-side (the progress-preview feed in CanvasWidgetView).
  },
  {
    // Legacy-parity "bbox overlay": dims everything OUTSIDE the generation frame.
    defaultValue: false,
    key: CANVAS_BBOX_OVERLAY_KEY,
    labelKey: 'widgets.canvas.settings.bboxOverlay',
    section: 'display',
    store: 'bboxOverlay',
  },
  {
    // webv2 extra (no legacy counterpart): hides the passive dashed bbox frame.
    defaultValue: true,
    key: CANVAS_SHOW_BBOX_KEY,
    labelKey: 'widgets.canvas.settings.showBbox',
    section: 'display',
    store: 'showBbox',
  },
  {
    defaultValue: true,
    key: CANVAS_CHECKERBOARD_KEY,
    labelKey: 'widgets.canvas.settings.checkerboard',
    section: 'display',
    store: 'checkerboard',
  },
  // ── Grid ──────────────────────────────────────────────────────────────────
  {
    defaultValue: false,
    key: CANVAS_SHOW_GRID_KEY,
    labelKey: 'widgets.canvas.settings.grid',
    section: 'grid',
    store: 'showGrid',
  },
  {
    defaultValue: true,
    key: CANVAS_SNAP_TO_GRID_KEY,
    labelKey: 'widgets.canvas.settings.snapToGrid',
    section: 'grid',
    store: 'snapToGrid',
  },
  {
    defaultValue: false,
    key: CANVAS_RULE_OF_THIRDS_KEY,
    labelKey: 'widgets.canvas.settings.ruleOfThirds',
    section: 'grid',
    store: 'ruleOfThirds',
  },
];

/** The sections, in popover render order. */
export const CANVAS_SETTING_SECTIONS: readonly CanvasSettingSection[] = ['behavior', 'display', 'grid'];

/** Resolved settings, keyed by persisted setting key (default-applied). */
export type ResolvedCanvasSettings = Record<string, boolean>;

/** Reads one boolean setting from a widget's `state.values`, applying its default when unset/non-boolean. */
export const readCanvasBooleanSetting = (
  values: Record<string, unknown> | undefined,
  setting: CanvasBooleanSetting
): boolean => {
  const raw = values?.[setting.key];
  return typeof raw === 'boolean' ? raw : setting.defaultValue;
};

/** Resolves every canvas setting from persisted values into a key→boolean map. */
export const resolveCanvasSettings = (values: Record<string, unknown> | undefined): ResolvedCanvasSettings => {
  const resolved: ResolvedCanvasSettings = {};
  for (const setting of CANVAS_SETTINGS) {
    resolved[setting.key] = readCanvasBooleanSetting(values, setting);
  }
  return resolved;
};

/** Structural equality for two resolved-settings maps (stable selector identity). */
export const canvasSettingsEqual = (a: ResolvedCanvasSettings, b: ResolvedCanvasSettings): boolean =>
  CANVAS_SETTINGS.every((setting) => a[setting.key] === b[setting.key]);
