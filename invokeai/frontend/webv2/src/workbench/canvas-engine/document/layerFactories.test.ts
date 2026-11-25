import { CONTROL_ADAPTER_DEFAULTS } from '@workbench/controlAdapters';
import { describe, expect, it } from 'vitest';

import {
  advanceMaskFillColors,
  createControlLayer,
  nextControlLayerName,
  nextMaskFillColor,
  resetMaskFillColors,
} from './layerFactories';

describe('createControlLayer', () => {
  it('creates an empty, untyped, model-less layer on the legacy ControlNet defaults with transparency on', () => {
    const layer = createControlLayer({ id: 'c1', name: 'Control Layer 1' });
    expect(layer).toEqual({
      adapter: { beginEndStepPct: [0, 0.75], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 0.75 },
      blendMode: 'normal',
      id: 'c1',
      isEnabled: true,
      isLocked: false,
      name: 'Control Layer 1',
      opacity: 1,
      source: { bitmap: null, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'control',
      withTransparencyEffect: true,
    });
    expect(layer.adapter.beginEndStepPct).not.toBe(CONTROL_ADAPTER_DEFAULTS.controlnet.beginEndStepPct);
  });

  it.each([
    ['anima', 'anima_lllite'],
    ['z-image', 'z_image_control'],
    ['sdxl', 'controlnet'],
  ] as const)('starts a %s layer on its kind’s defaults without a model', (base, kind) => {
    expect(createControlLayer({ base, id: 'c1', name: 'Control Layer 1' }).adapter).toEqual({
      ...CONTROL_ADAPTER_DEFAULTS[kind],
      model: null,
    });
  });

  it('uses the given pixel source', () => {
    const source = { image: { height: 48, imageName: 'photo.png', width: 64 }, type: 'image' } as const;
    expect(createControlLayer({ id: 'c1', name: 'C', source }).source).toBe(source);
  });
});

describe('nextControlLayerName', () => {
  it('returns the first free "Control Layer N" gap', () => {
    expect(nextControlLayerName([])).toBe('Control Layer 1');
    expect(nextControlLayerName(['Control Layer 1', 'Control Layer 3', 'Layer 2'])).toBe('Control Layer 2');
    expect(nextControlLayerName(['Control Layer 1', 'Control Layer 2'])).toBe('Control Layer 3');
  });
});

it('cycles mask colors independently in palette order and restarts at the session defaults', () => {
  resetMaskFillColors();
  expect(nextMaskFillColor('inpaint_mask')).toBe('#83d683');
  expect(nextMaskFillColor('regional_guidance')).toBe('#83d683');
  expect(Array.from({ length: 10 }, (_, index) => nextMaskFillColor('inpaint_mask', index))).toEqual([
    '#83d683',
    '#F59E42',
    '#F4D35E',
    '#38C9C5',
    '#62B5F5',
    '#637BE8',
    '#AD80E8',
    '#ED79BA',
    '#B7D65A',
    '#e07575',
  ]);
  advanceMaskFillColors('inpaint_mask');
  expect(nextMaskFillColor('inpaint_mask')).toBe('#F59E42');
  expect(nextMaskFillColor('regional_guidance')).toBe('#83d683');
  advanceMaskFillColors('regional_guidance');
  expect(nextMaskFillColor('regional_guidance')).toBe('#F59E42');
  resetMaskFillColors();
  expect(nextMaskFillColor('regional_guidance')).toBe('#83d683');
  expect(nextMaskFillColor('inpaint_mask')).toBe('#83d683');
});
