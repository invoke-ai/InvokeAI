import type { ModelConfig } from '@features/models';
import type { CanvasControlAdapterContract } from '@workbench/canvas-engine/api';

import { CONTROL_ADAPTER_DEFAULTS } from '@workbench/controlAdapters';
import { describe, expect, it } from 'vitest';

import {
  getCompatibleControlModels,
  getReidentifyPendingControlModels,
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

  it('returns only Qwen-Image ControlNet models for qwen_image_control', () => {
    const models = [
      model('qwen-union', 'qwen-image', 'controlnet'),
      model('z-control', 'z-image', 'controlnet'),
      model('qwen-lora', 'qwen-image', 'lora'),
    ];

    expect(
      getCompatibleControlModels(models, 'qwen-image', 'qwen_image_control').map((candidate) => candidate.key)
    ).toEqual(['qwen-union']);
    expect(getCompatibleControlModels(models, 'z-image', 'qwen_image_control')).toEqual([]);
    expect(getCompatibleControlModels(models, null, 'qwen_image_control')).toEqual([]);
  });

  it('returns only FLUX.2 ControlNet models for flux2_control', () => {
    const models = [
      model('flux2-union', 'flux2', 'controlnet'),
      model('flux-union', 'flux', 'controlnet'),
      model('flux2-lora', 'flux2', 'lora'),
    ];

    expect(getCompatibleControlModels(models, 'flux2', 'flux2_control').map((candidate) => candidate.key)).toEqual([
      'flux2-union',
    ]);
    expect(getCompatibleControlModels(models, 'flux', 'flux2_control')).toEqual([]);
    expect(getCompatibleControlModels(models, null, 'flux2_control')).toEqual([]);
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

  it('lists only Anima adapters of unknown type as needing re-identify, for the LLLite kind', () => {
    const others = [
      ...models,
      { ...model('sdxl-unknown', 'sdxl', 'controlnet'), cond_in_channels: null } as ModelConfig,
      { ...model('anima-lora', 'anima', 'lora'), cond_in_channels: null } as ModelConfig,
    ];

    expect(
      getReidentifyPendingControlModels(others, 'anima', 'anima_lllite').map((candidate) => candidate.key)
    ).toEqual(['unidentified']);
    expect(getReidentifyPendingControlModels(others, 'anima', 'controlnet')).toEqual([]);
    expect(getReidentifyPendingControlModels(others, 'sdxl', 'anima_lllite')).toEqual([]);
  });

  it('never offers an adapter that needs re-identifying', () => {
    expect(
      getCompatibleControlModels([anima('unidentified', null), anima('inpainting', 4)], 'anima', 'anima_lllite')
    ).toEqual([]);
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
    expect(switchControlAdapterKind(negative, 'qwen_image_control', [], 'qwen-image')).toMatchObject({
      beginEndStepPct: [0, 0.75],
      weight: CONTROL_ADAPTER_DEFAULTS.qwen_image_control.weight,
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
