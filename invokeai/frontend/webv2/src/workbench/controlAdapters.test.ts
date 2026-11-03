import { describe, expect, it } from 'vitest';

import {
  CONTROL_ADAPTER_DEFAULTS,
  createDefaultControlAdapter,
  deriveControlKind,
  getDefaultControlAdapterKind,
  normalizeControlAdapter,
} from './controlAdapters';

describe('normalizeControlAdapter', () => {
  it.each([
    ['controlnet', -1, [0, 0.75]],
    ['t2i_adapter', 2, [0.2, 0.8]],
    ['control_lora', -1, [0, 1]],
    ['z_image_control', 0, [0.1, 0.9]],
    ['anima_lllite', -1, [0.1, 0.9]],
    ['qwen_image_control', 0, [0.1, 0.9]],
    ['flux2_control', 0, [0.1, 0.9]],
  ] as const)('preserves valid %s numeric values', (kind, weight, beginEndStepPct) => {
    const adapter = { ...CONTROL_ADAPTER_DEFAULTS[kind], beginEndStepPct, weight };
    expect(normalizeControlAdapter(adapter)).toEqual(adapter);
  });

  it.each([null, 'z-image-union'])(
    'keeps the mode and a negative weight whatever kind is stored (model %s): the resolved kind governs',
    (model) => {
      // Created under Z-Image, now resolved to an SDXL ControlNet; a stored pick may belong to the other base.
      const adapter = {
        beginEndStepPct: [0.1, 0.8],
        controlMode: 'more_control',
        kind: 'z_image_control',
        model,
        weight: -0.5,
      };
      expect(normalizeControlAdapter(adapter)).toEqual(adapter);
    }
  );

  it.each([
    ['controlnet', Number.NaN],
    ['controlnet', Number.POSITIVE_INFINITY],
    ['controlnet', -1.01],
    ['controlnet', 2.01],
    ['z_image_control', -1.5],
    ['z_image_control', 2.01],
    ['anima_lllite', 2.01],
    ['qwen_image_control', -1.01],
    ['flux2_control', 2.01],
  ] as const)('repairs a %s weight of %s, outside what any kind runs, with its default', (kind, weight) => {
    const adapter = { ...CONTROL_ADAPTER_DEFAULTS[kind], model: 'm', weight };
    expect(normalizeControlAdapter(adapter)).toMatchObject({ kind, weight: CONTROL_ADAPTER_DEFAULTS[kind].weight });
  });

  it('maps the contract fields only, and repairs an unknown mode with the kind default', () => {
    const persisted = {
      beginEndStepPct: [0.1, 0.9],
      controlMode: 'sideways',
      extra: true,
      kind: 'controlnet',
      model: 7,
      weight: 1.5,
    };
    expect(normalizeControlAdapter(persisted)).toEqual({
      beginEndStepPct: [0.1, 0.9],
      controlMode: 'balanced',
      kind: 'controlnet',
      model: null,
      weight: 1.5,
    });
  });

  it.each([
    [Number.NaN, 1],
    [0, Number.POSITIVE_INFINITY],
    [-0.01, 1],
    [0, 1.01],
    [0.5, 0.5],
    [0.8, 0.2],
  ])('repairs invalid step range [%s, %s] with adapter defaults', (begin, end) => {
    const adapter = { ...CONTROL_ADAPTER_DEFAULTS.z_image_control, beginEndStepPct: [begin, end] };
    expect(normalizeControlAdapter(adapter)).toMatchObject({ beginEndStepPct: [0, 1] });
  });
});

describe('default control adapters', () => {
  it('starts each base on the one kind it accepts', () => {
    expect(getDefaultControlAdapterKind('anima')).toBe('anima_lllite');
    expect(getDefaultControlAdapterKind('z-image')).toBe('z_image_control');
    expect(getDefaultControlAdapterKind('qwen-image')).toBe('qwen_image_control');
    expect(getDefaultControlAdapterKind('flux2')).toBe('flux2_control');
    expect(getDefaultControlAdapterKind('sdxl')).toBe('controlnet');
    expect(getDefaultControlAdapterKind(null)).toBe('controlnet');
  });

  it('starts a Qwen-Image control at the union model card recommendation', () => {
    expect(createDefaultControlAdapter('qwen-image', 'union')).toEqual({
      beginEndStepPct: [0, 1],
      controlMode: null,
      kind: 'qwen_image_control',
      model: 'union',
      weight: 0.85,
    });
  });

  it('starts a FLUX.2 control at the union model card recommendation', () => {
    expect(createDefaultControlAdapter('flux2', 'union')).toEqual({
      beginEndStepPct: [0, 1],
      controlMode: null,
      kind: 'flux2_control',
      model: 'union',
      weight: 0.75,
    });
  });

  it('hands out a step range the caller may mutate without touching the defaults', () => {
    const adapter = createDefaultControlAdapter('anima');
    adapter.beginEndStepPct[0] = 0.5;
    expect(CONTROL_ADAPTER_DEFAULTS.anima_lllite.beginEndStepPct).toEqual([0, 1]);
  });
});

describe('deriveControlKind', () => {
  it.each([
    [{ base: 'sdxl', type: 'controlnet' }, 'controlnet'],
    [{ base: 'flux', type: 'controlnet' }, 'controlnet'],
    [{ base: 'sdxl', type: 't2i_adapter' }, 't2i_adapter'],
    [{ base: 'flux', type: 'control_lora' }, 'control_lora'],
    [{ base: 'z-image', type: 'controlnet' }, 'z_image_control'],
    [{ base: 'qwen-image', type: 'controlnet' }, 'qwen_image_control'],
    [{ base: 'flux2', type: 'controlnet' }, 'flux2_control'],
    [{ base: 'anima', type: 'controlnet' }, 'anima_lllite'],
    [{ base: 'sdxl', type: 'main' }, null],
    [{ base: 'sdxl', type: 'ip_adapter' }, null],
  ] as const)('runs %j as %s', (model, kind) => {
    expect(deriveControlKind(model)).toBe(kind);
  });
});
