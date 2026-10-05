import type { ModelConfig } from '@features/models';
import type { CanvasControlAdapterContract } from '@workbench/canvas-engine/api';

import { CONTROL_ADAPTER_DEFAULTS } from '@workbench/controlAdapters';
import { describe, expect, it } from 'vitest';

import {
  getCompatibleControlModels,
  resolveDefaultControlModel,
  resolveDefaultControlModelForBase,
  switchControlAdapterKind,
} from './controlModelOptions';

const model = (key: string, base: string, type: string): ModelConfig => ({ base, key, name: key, type }) as ModelConfig;

describe('getCompatibleControlModels', () => {
  it('returns only Z-Image ControlNet models for z_image_control', () => {
    const models = [
      model('z-control', 'z-image', 'controlnet'),
      model('sd-control', 'sd-1', 'controlnet'),
      model('z-t2i', 'z-image', 't2i_adapter'),
    ];

    expect(getCompatibleControlModels(models, 'z-image', 'z_image_control').map((candidate) => candidate.key)).toEqual([
      'z-control',
    ]);
    expect(getCompatibleControlModels(models, 'sd-1', 'z_image_control')).toEqual([]);
  });

  it('keeps existing adapter filtering unchanged', () => {
    const models = [model('sd-control', 'sd-1', 'controlnet'), model('flux-lora', 'flux', 'control_lora')];
    expect(getCompatibleControlModels(models, 'sd-1', 'controlnet').map((candidate) => candidate.key)).toEqual([
      'sd-control',
    ]);
    expect(getCompatibleControlModels(models, null, 'control_lora').map((candidate) => candidate.key)).toEqual([
      'flux-lora',
    ]);
  });
});

describe('getCompatibleControlModels — Anima ControlNet-LLLite', () => {
  const anima = (key: string, cond_in_channels: number | null) =>
    ({ ...model(key, 'anima', 'controlnet'), cond_in_channels }) as ModelConfig;
  const models = [
    anima('sketch', 3),
    anima('inpainting', 4),
    anima('unidentified', null),
    { ...model('sdxl-control', 'sdxl', 'controlnet'), cond_in_channels: 3 } as ModelConfig,
  ];

  it('offers only Anima control (3-channel) adapters, and only for an Anima main model', () => {
    expect(getCompatibleControlModels(models, 'anima', 'anima_lllite').map((candidate) => candidate.key)).toEqual([
      'sketch',
    ]);
    expect(getCompatibleControlModels(models, 'sdxl', 'anima_lllite')).toEqual([]);
    expect(getCompatibleControlModels(models, null, 'anima_lllite')).toEqual([]);
  });

  it('defaults a new Anima control layer to an LLLite model', () => {
    expect(resolveDefaultControlModelForBase(models, 'anima')).toBe('sketch');
  });
});

describe('switchControlAdapterKind', () => {
  const sketch = { ...model('sketch', 'anima', 'controlnet'), cond_in_channels: 3 } as ModelConfig;
  const inpainting = { ...model('inpainting', 'anima', 'controlnet'), cond_in_channels: 4 } as ModelConfig;
  const savedAsControlNet = (key: string): CanvasControlAdapterContract => ({
    beginEndStepPct: [0, 0.75],
    controlMode: 'balanced',
    kind: 'controlnet',
    model: key,
    weight: 0.75,
  });

  it('keeps the LLLite model, weight and step range of an Anima layer saved under ControlNet', () => {
    expect(switchControlAdapterKind(savedAsControlNet('sketch'), 'anima_lllite', [sketch], 'anima')).toEqual({
      beginEndStepPct: [0, 0.75],
      controlMode: null,
      kind: 'anima_lllite',
      model: 'sketch',
      weight: 0.75,
    });
  });

  it('takes the new kind’s default for each value it cannot use', () => {
    const negative = { ...savedAsControlNet('sketch'), weight: -0.5 };
    expect(switchControlAdapterKind(negative, 'z_image_control', [], 'z-image')).toMatchObject({
      beginEndStepPct: [0, 0.75],
      weight: CONTROL_ADAPTER_DEFAULTS.z_image_control.weight,
    });
    const unordered = { ...savedAsControlNet('sketch'), beginEndStepPct: [0.6, 0.2] as [number, number] };
    expect(switchControlAdapterKind(unordered, 'anima_lllite', [sketch], 'anima')).toMatchObject({
      beginEndStepPct: CONTROL_ADAPTER_DEFAULTS.anima_lllite.beginEndStepPct,
      weight: 0.75,
    });
  });

  it('clears a model the new kind cannot run', () => {
    expect(
      switchControlAdapterKind(savedAsControlNet('inpainting'), 'anima_lllite', [inpainting], 'anima').model
    ).toBeNull();
    const sdxl = model('sdxl-control', 'sdxl', 'controlnet');
    expect(switchControlAdapterKind(savedAsControlNet('sdxl-control'), 't2i_adapter', [sdxl], 'sdxl').model).toBeNull();
  });
});

describe('resolveDefaultControlModel', () => {
  it('prefers a union model, then tile, then the first compatible', () => {
    const canny = model('canny', 'sdxl', 'controlnet');
    const tile = { ...model('tile-model', 'sdxl', 'controlnet'), name: 'ControlNet Tile' } as ModelConfig;
    const union = { ...model('union-model', 'sdxl', 'controlnet'), name: 'Union Pro XL' } as ModelConfig;

    expect(resolveDefaultControlModel([canny, tile, union], 'sdxl', 'controlnet')).toBe('union-model');
    expect(resolveDefaultControlModel([canny, tile], 'sdxl', 'controlnet')).toBe('tile-model');
    expect(resolveDefaultControlModel([canny], 'sdxl', 'controlnet')).toBe('canny');
  });

  it('only considers models compatible with the base and kind', () => {
    const models = [
      { ...model('sd1-union', 'sd-1', 'controlnet'), name: 'Union' } as ModelConfig,
      model('sdxl-canny', 'sdxl', 'controlnet'),
    ];

    expect(resolveDefaultControlModel(models, 'sdxl', 'controlnet')).toBe('sdxl-canny');
    expect(resolveDefaultControlModel(models, 'sdxl', 't2i_adapter')).toBeNull();
    expect(resolveDefaultControlModel([], 'sdxl', 'controlnet')).toBeNull();
  });
});

describe('resolveDefaultControlModelForBase', () => {
  it('resolves the z_image_control kind for the z-image base and controlnet otherwise', () => {
    const models = [model('z-control', 'z-image', 'controlnet'), model('sd-control', 'sd-1', 'controlnet')];

    expect(resolveDefaultControlModelForBase(models, 'z-image')).toBe('z-control');
    expect(resolveDefaultControlModelForBase(models, 'sd-1')).toBe('sd-control');
    expect(resolveDefaultControlModelForBase(models, 'sd-2')).toBeNull();
  });
});
