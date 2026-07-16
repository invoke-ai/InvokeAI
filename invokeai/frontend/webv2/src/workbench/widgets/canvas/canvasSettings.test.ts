import { describe, expect, it } from 'vitest';

import {
  CANVAS_BBOX_OVERLAY_KEY,
  CANVAS_CHECKERBOARD_KEY,
  CANVAS_INVERT_BRUSH_SCROLL_KEY,
  CANVAS_RULE_OF_THIRDS_KEY,
  CANVAS_SETTING_SECTIONS,
  CANVAS_SETTINGS,
  CANVAS_SHOW_BBOX_KEY,
  CANVAS_SHOW_GRID_KEY,
  CANVAS_SHOW_PROGRESS_KEY,
  CANVAS_SNAP_TO_GRID_KEY,
  canvasSettingsEqual,
  readCanvasBooleanSetting,
  resolveCanvasSettings,
} from './canvasSettings';

const settingByKey = (key: string) => {
  const found = CANVAS_SETTINGS.find((s) => s.key === key);
  if (!found) {
    throw new Error(`no setting for key ${key}`);
  }
  return found;
};

const DEFAULTS = {
  [CANVAS_BBOX_OVERLAY_KEY]: false,
  [CANVAS_CHECKERBOARD_KEY]: true,
  [CANVAS_INVERT_BRUSH_SCROLL_KEY]: false,
  [CANVAS_RULE_OF_THIRDS_KEY]: false,
  [CANVAS_SHOW_BBOX_KEY]: true,
  [CANVAS_SHOW_GRID_KEY]: false,
  [CANVAS_SHOW_PROGRESS_KEY]: true,
  [CANVAS_SNAP_TO_GRID_KEY]: true,
};

describe('canvasSettings persistence mapping', () => {
  it('applies each setting default when values are missing', () => {
    expect(resolveCanvasSettings(undefined)).toEqual(DEFAULTS);
    expect(resolveCanvasSettings({})).toEqual(DEFAULTS);
  });

  it('reads persisted booleans over the defaults', () => {
    const resolved = resolveCanvasSettings({
      [CANVAS_BBOX_OVERLAY_KEY]: true,
      [CANVAS_CHECKERBOARD_KEY]: false,
      [CANVAS_INVERT_BRUSH_SCROLL_KEY]: true,
      [CANVAS_RULE_OF_THIRDS_KEY]: true,
      [CANVAS_SHOW_BBOX_KEY]: false,
      [CANVAS_SHOW_GRID_KEY]: true,
      [CANVAS_SHOW_PROGRESS_KEY]: false,
      [CANVAS_SNAP_TO_GRID_KEY]: false,
    });
    expect(resolved).toEqual({
      [CANVAS_BBOX_OVERLAY_KEY]: true,
      [CANVAS_CHECKERBOARD_KEY]: false,
      [CANVAS_INVERT_BRUSH_SCROLL_KEY]: true,
      [CANVAS_RULE_OF_THIRDS_KEY]: true,
      [CANVAS_SHOW_BBOX_KEY]: false,
      [CANVAS_SHOW_GRID_KEY]: true,
      [CANVAS_SHOW_PROGRESS_KEY]: false,
      [CANVAS_SNAP_TO_GRID_KEY]: false,
    });
  });

  it('falls back to the default for a non-boolean persisted value', () => {
    expect(readCanvasBooleanSetting({ [CANVAS_CHECKERBOARD_KEY]: 'nope' }, settingByKey(CANVAS_CHECKERBOARD_KEY))).toBe(
      true
    );
    expect(readCanvasBooleanSetting({ [CANVAS_SHOW_GRID_KEY]: 1 }, settingByKey(CANVAS_SHOW_GRID_KEY))).toBe(false);
  });

  it('maps every setting to a unique key and a known section', () => {
    const keys = CANVAS_SETTINGS.map((s) => s.key);
    expect(new Set(keys).size).toBe(keys.length);
    for (const setting of CANVAS_SETTINGS) {
      expect(CANVAS_SETTING_SECTIONS).toContain(setting.section);
    }
  });

  it('gives every engine-backed store a distinct store id (React-consumed settings have none)', () => {
    const stores = CANVAS_SETTINGS.map((s) => s.store).filter((s): s is NonNullable<typeof s> => s !== undefined);
    expect(new Set(stores).size).toBe(stores.length);
    // showProgressOnCanvas is consumed React-side and intentionally has no store.
    expect(settingByKey(CANVAS_SHOW_PROGRESS_KEY).store).toBeUndefined();
  });
});

describe('canvasSettingsEqual', () => {
  it('is true for equal maps and false when any setting differs', () => {
    const base = resolveCanvasSettings({});
    expect(canvasSettingsEqual(base, resolveCanvasSettings({}))).toBe(true);
    expect(canvasSettingsEqual(base, resolveCanvasSettings({ [CANVAS_SHOW_GRID_KEY]: true }))).toBe(false);
    expect(canvasSettingsEqual(base, resolveCanvasSettings({ [CANVAS_SHOW_BBOX_KEY]: false }))).toBe(false);
  });
});
