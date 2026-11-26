import { layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
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
    '#f59e42',
    '#f4d35e',
    '#38c9c5',
    '#62b5f5',
    '#637be8',
    '#ad80e8',
    '#ed79ba',
    '#b7d65a',
    '#e07575',
  ]);
  advanceMaskFillColors('inpaint_mask');
  expect(nextMaskFillColor('inpaint_mask')).toBe('#f59e42');
  expect(nextMaskFillColor('regional_guidance')).toBe('#83d683');
  advanceMaskFillColors('regional_guidance');
  expect(nextMaskFillColor('regional_guidance')).toBe('#f59e42');
  resetMaskFillColors();
  expect(nextMaskFillColor('regional_guidance')).toBe('#83d683');
  expect(nextMaskFillColor('inpaint_mask')).toBe('#83d683');
});

it.each([
  { label: 'same type and uppercase hex', used: ['#83D683'], other: [], expected: ['#f59e42'] },
  { label: 'colors used by the other mask type', used: [], other: ['#83d683'], expected: ['#83d683'] },
  { label: 'an import batch', used: ['#83d683', '#f4d35e'], other: [], expected: ['#f59e42', '#38c9c5', '#62b5f5'] },
  {
    label: 'an exhausted palette',
    used: [
      '#e07575',
      '#83d683',
      '#f59e42',
      '#f4d35e',
      '#38c9c5',
      '#62b5f5',
      '#637be8',
      '#ad80e8',
      '#ed79ba',
      '#b7d65a',
    ],
    other: [],
    expected: ['#83d683', '#f59e42'],
  },
])('chooses palette colors with $label', ({ used, other, expected }) => {
  resetMaskFillColors();
  const layers = [
    ...used.map((color) =>
      layerContract(color, 'regional_guidance', { mask: { bitmap: null, fill: { color, style: 'solid' } } })
    ),
    ...other.map((color) =>
      layerContract(color, 'inpaint_mask', { mask: { bitmap: null, fill: { color, style: 'solid' } } })
    ),
  ];
  expect(expected.map((_, offset) => nextMaskFillColor('regional_guidance', offset, layers))).toEqual(expected);
});
