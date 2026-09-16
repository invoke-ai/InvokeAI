import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import { describe, expect, it } from 'vitest';
import type { z } from 'zod';

import {
  canvasSettingsSliceConfig,
  settingsPressureAffectsOpacityToggled,
  settingsPressureAffectsWidthToggled,
  settingsTraceTaperEndsToggled,
} from './canvasSettingsSlice';

describe('canvasSettingsSlice', () => {
  type InitialState = ReturnType<typeof canvasSettingsSliceConfig.getInitialState>;
  type SchemaState = z.infer<typeof canvasSettingsSliceConfig.schema>;

  const { reducer } = canvasSettingsSliceConfig.slice;
  const migrate = canvasSettingsSliceConfig.persistConfig?.migrate;

  it('keeps the initial state aligned with the persisted schema', () => {
    assert<Equals<InitialState, SchemaState>>();
  });

  it('toggles pressure-width and pressure-opacity independently', () => {
    const state = canvasSettingsSliceConfig.getInitialState();

    const pressureWidthDisabled = reducer(state, settingsPressureAffectsWidthToggled());
    const pressureOpacityEnabled = reducer(pressureWidthDisabled, settingsPressureAffectsOpacityToggled());

    expect(pressureWidthDisabled.pressureAffectsWidth).toBe(false);
    expect(pressureWidthDisabled.pressureAffectsOpacity).toBe(false);
    expect(pressureOpacityEnabled.pressureAffectsWidth).toBe(false);
    expect(pressureOpacityEnabled.pressureAffectsOpacity).toBe(true);
  });

  it('toggles tapered vector tracing independently', () => {
    const state = canvasSettingsSliceConfig.getInitialState();

    const result = reducer(state, settingsTraceTaperEndsToggled());

    expect(state.traceTaperEnds).toBe(false);
    expect(result.traceTaperEnds).toBe(true);
  });

  it('defaults tapered vector tracing off when migrating older settings', () => {
    expect(migrate).toBeDefined();
    const { traceTaperEnds: _traceTaperEnds, ...olderState } = canvasSettingsSliceConfig.getInitialState();

    const result = migrate?.(olderState) as InitialState;

    expect(result.traceTaperEnds).toBe(false);
  });

  it('migrates legacy pressureSensitivity to pressureAffectsWidth and leaves opacity disabled', () => {
    expect(migrate).toBeDefined();

    const result = migrate?.({
      ...canvasSettingsSliceConfig.getInitialState(),
      pressureSensitivity: true,
      pressureAffectsWidth: undefined,
      pressureAffectsOpacity: undefined,
    }) as InitialState;

    expect(result.pressureAffectsWidth).toBe(true);
    expect(result.pressureAffectsOpacity).toBe(false);
  });

  it('preserves explicit split pressure settings during migration', () => {
    expect(migrate).toBeDefined();

    const result = migrate?.({
      ...canvasSettingsSliceConfig.getInitialState(),
      pressureAffectsWidth: false,
      pressureAffectsOpacity: true,
    }) as InitialState;

    expect(result.pressureAffectsWidth).toBe(false);
    expect(result.pressureAffectsOpacity).toBe(true);
  });
});
