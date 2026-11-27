import type { ControlAdapterKind } from '@features/generation/graph';
import type { ModelConfig } from '@features/models';
import type { CanvasControlLayerContract, ControlType } from '@workbench/canvas-engine/api';

import { describe, expect, it } from 'vitest';

import {
  getTypeAvailability,
  isRememberableControlPick,
  resolveControlModel,
  type ControlModelMemory,
} from './controlModelResolution';

const model = (key: string, base: string, type = 'controlnet', extra: Partial<ModelConfig> = {}): ModelConfig =>
  ({ base, key, name: key, type, ...extra }) as ModelConfig;

type Layer = Pick<CanvasControlLayerContract, 'controlType' | 'detection' | 'adapter'>;

const layer = (
  controlType: ControlType | undefined,
  overrides: { detector?: string | null; model?: string | null; kind?: ControlAdapterKind } = {}
): Layer => ({
  adapter: {
    beginEndStepPct: [0, 1],
    controlMode: null,
    kind: overrides.kind ?? 'controlnet',
    model: overrides.model ?? null,
    weight: 0.75,
  },
  ...(controlType ? { controlType } : {}),
  ...(overrides.detector === undefined
    ? {}
    : { detection: overrides.detector === null ? null : { settings: {}, type: overrides.detector } }),
});

const supportsAll = (): boolean => true;

const resolve = (
  target: Layer,
  base: string | null,
  models: readonly ModelConfig[],
  remembered: ControlModelMemory = {},
  supportsKind: (kind: ControlAdapterKind) => boolean = supportsAll
) => resolveControlModel({ base, layer: target, models, remembered, supportsKind });

const depthA = model('sdxl-depth-a', 'sdxl');
const depthB = model('sdxl-depth-b', 'sdxl', 'controlnet', {
  default_settings: { preprocessor: 'depth_anything_image_processor' },
});
const normal = model('sdxl-normal', 'sdxl');
const pose = model('sdxl-openpose', 'sdxl');
const sdxlUnion = model('xinsir-controlnet-union-sdxl-promax', 'sdxl');
const sd1Depth = model('sd15-depth', 'sd-1');

describe('resolveControlModel without a type', () => {
  it('applies the stored adapter as it is', () => {
    const resolution = resolve(layer(undefined, { kind: 't2i_adapter', model: pose.key }), 'sdxl', [pose]);

    expect(resolution).toMatchObject({ how: 'none', kind: 't2i_adapter', model: pose, unresolved: null });
    expect(resolution.candidates).toEqual([]);
  });

  it('lists every model it could run on the base, for a direct pick, and no candidates', () => {
    const resolution = resolve(layer(undefined), 'sdxl', [
      depthA,
      pose,
      sdxlUnion,
      sd1Depth,
      model('sdxl-main', 'sdxl', 'main'),
      model('my-sdxl-control', 'sdxl'),
    ]);

    expect(resolution.usable.map((entry) => entry.key)).toEqual([
      depthA.key,
      pose.key,
      sdxlUnion.key,
      'my-sdxl-control',
    ]);
    expect(resolution.candidates).toEqual([]);
  });

  it('reports a missing type when the layer names no model', () => {
    expect(resolve(layer(undefined), 'sdxl', [pose])).toMatchObject({
      how: 'none',
      model: null,
      unresolved: 'missing_type',
    });
  });

  it('reports a missing model, not a missing type, when an untyped layer names one that is not installed', () => {
    expect(resolve(layer(undefined, { model: 'deleted' }), 'sdxl', [pose])).toMatchObject({
      how: 'none',
      model: null,
      unresolved: 'missing_model',
    });
  });
});

describe('resolveControlModel with a type', () => {
  it('runs an explicit pick for this base as the kind that model runs', () => {
    const t2i = model('t2i-depth', 'sdxl', 't2i_adapter');
    const resolution = resolve(layer('depth', { model: t2i.key }), 'sdxl', [depthA, t2i]);

    expect(resolution).toMatchObject({ how: 'explicit', kind: 't2i_adapter', model: t2i, unresolved: null });
  });

  it('ignores a pick made for another base, and one that was deleted', () => {
    expect(resolve(layer('depth', { model: sd1Depth.key }), 'sdxl', [sd1Depth, depthA])).toMatchObject({
      how: 'only',
      model: depthA,
    });
    expect(resolve(layer('depth', { model: 'deleted' }), 'sdxl', [depthA])).toMatchObject({
      how: 'only',
      model: depthA,
    });
  });

  it('runs the only model made for the type', () => {
    expect(resolve(layer('pose'), 'sdxl', [depthA, pose, sdxlUnion])).toMatchObject({
      how: 'only',
      kind: 'controlnet',
      model: pose,
    });
  });

  it('asks when several models fit, listing those made for the type first, then unions', () => {
    const fluxDepth = model('flux-depth-lora', 'flux', 'control_lora');
    const fluxUnion = model('flux-union-pro', 'flux');
    const fluxUnion2 = model('flux-union-alpha', 'flux');

    expect(resolve(layer('depth'), 'sdxl', [depthA, depthB])).toMatchObject({
      candidates: [depthA, depthB],
      how: 'ask',
      model: null,
      unresolved: 'choose_model',
    });
    expect(resolve(layer('depth'), 'flux', [fluxUnion, fluxDepth]).candidates).toEqual([fluxDepth, fluxUnion]);
    expect(resolve(layer('edges'), 'flux', [fluxUnion, fluxUnion2])).toMatchObject({
      candidates: [fluxUnion, fluxUnion2],
      how: 'ask',
      unresolved: 'choose_model',
    });
  });

  it('settles a choice with the remembered pick for this base and type, only while it is a candidate', () => {
    const models = [depthA, depthB, pose];

    expect(resolve(layer('depth'), 'sdxl', models, { 'sdxl:depth': depthB.key })).toMatchObject({
      how: 'remembered',
      model: depthB,
    });
    expect(resolve(layer('depth'), 'sdxl', models, { 'sdxl:depth': pose.key }).how).toBe('ask');
    expect(resolve(layer('depth'), 'sdxl', models, { 'sdxl:depth': 'deleted' }).how).toBe('ask');
    expect(resolve(layer('depth'), 'sdxl', models, { 'sd-1:depth': depthB.key }).how).toBe('ask');
  });

  it('prefers the remembered covering union over the only model made for the type', () => {
    const models = [depthA, sdxlUnion];

    expect(resolve(layer('depth'), 'sdxl', models)).toMatchObject({ how: 'only', model: depthA });
    expect(resolve(layer('depth'), 'sdxl', models, { 'sdxl:depth': sdxlUnion.key })).toMatchObject({
      how: 'remembered',
      model: sdxlUnion,
    });
  });

  it('matches the family the chosen detector produces', () => {
    const canny = model('sdxl-canny', 'sdxl');
    const mlsd = model('sdxl-mlsd', 'sdxl');

    expect(
      resolve(layer('depth', { detector: 'depth_anything_depth_estimation' }), 'sdxl', [depthA, normal]).model
    ).toBe(depthA);
    expect(resolve(layer('depth', { detector: 'normal_map' }), 'sdxl', [depthA, normal]).model).toBe(normal);
    expect(resolve(layer('edges', { detector: 'mlsd_detection' }), 'sdxl', [canny, mlsd]).model).toBe(mlsd);
    expect(resolve(layer('edges', { detector: null }), 'sdxl', [canny, mlsd]).model).toBe(canny);
  });

  it('falls back to a model made for the type when none matches the family', () => {
    expect(resolve(layer('depth'), 'sdxl', [normal])).toMatchObject({ how: 'only', model: normal });
  });

  it('uses the one union model that covers the family', () => {
    const zUnion = model('z-image-fun-controlnet', 'z-image');

    expect(resolve(layer('pose'), 'sdxl', [sdxlUnion])).toMatchObject({ how: 'union', model: sdxlUnion });
    expect(resolve(layer('color'), 'sdxl', [sdxlUnion]).unresolved).toBe('no_model_for_type');

    expect(resolve(layer('depth'), 'z-image', [zUnion])).toMatchObject({
      how: 'union',
      kind: 'z_image_control',
      model: zUnion,
    });
    expect(resolve(layer('lineart'), 'z-image', [zUnion])).toMatchObject({
      how: 'missing',
      kind: 'z_image_control',
      unresolved: 'no_model_for_type',
    });
  });

  it('resolves a Qwen-Image InstantX union only for the families it was trained on', () => {
    const instantx = model('qwen-instantx-union', 'qwen-image', 'controlnet', {
      control_architecture: 'instantx_union',
    });

    expect(resolve(layer('edges', { detector: 'canny_edge_detection' }), 'qwen-image', [instantx])).toMatchObject({
      how: 'union',
      kind: 'qwen_image_control',
      model: instantx,
    });
    expect(resolve(layer('edges', { detector: 'mlsd_detection' }), 'qwen-image', [instantx])).toMatchObject({
      how: 'missing',
      model: null,
      unresolved: 'no_model_for_type',
    });
  });

  it('tells no control models apart from none for this type', () => {
    expect(resolve(layer('pose'), 'sdxl', [depthA, sd1Depth])).toMatchObject({
      how: 'missing',
      unresolved: 'no_model_for_type',
    });
    expect(resolve(layer('pose'), 'sdxl', [sd1Depth])).toMatchObject({
      how: 'missing',
      kind: 'controlnet',
      unresolved: 'no_control_models',
    });
    // An SD1.5 union carries no coverage until the backend reports its control types.
    expect(resolve(layer('pose'), 'sd-1', [model('controlnet-union-sd15', 'sd-1')]).unresolved).toBe(
      'no_model_for_type'
    );
  });

  it('only counts models the base runs and validation would accept', () => {
    const t2i = model('t2i-depth', 'sdxl', 't2i_adapter');
    const noT2i = (kind: ControlAdapterKind) => kind !== 't2i_adapter';
    const lllite = model('anima-lllite-depth', 'anima', 'controlnet', { cond_in_channels: 3 });
    const inpaint = model('anima-lllite-depth-inpaint', 'anima', 'controlnet', { cond_in_channels: 4 });

    expect(resolve(layer('depth'), 'sdxl', [t2i], {}, noT2i).unresolved).toBe('no_control_models');
    expect(resolve(layer('depth', { model: t2i.key }), 'sdxl', [t2i, depthA], {}, noT2i).model).toBe(depthA);
    expect(resolve(layer('depth'), 'anima', [lllite, inpaint])).toMatchObject({
      how: 'only',
      kind: 'anima_lllite',
      model: lllite,
    });
  });

  it('considers every base when no main model is selected', () => {
    expect(resolve(layer('depth'), null, [sd1Depth]).model).toBe(sd1Depth);
  });
});

describe('resolveControlModel model lists and family match', () => {
  it('lists every model the layer could run on this base, beside the candidates for its type', () => {
    const resolution = resolve(layer('depth'), 'sdxl', [
      depthA,
      pose,
      sdxlUnion,
      sd1Depth,
      model('sdxl-main', 'sdxl', 'main'),
    ]);

    expect(resolution.candidates).toEqual([depthA, sdxlUnion]);
    expect(resolution.usable).toEqual([depthA, pose, sdxlUnion]);
  });

  it('says whether the model matches the detector family or only the type', () => {
    const fluxUnion = model('flux-union-pro', 'flux');

    expect(resolve(layer('depth'), 'sdxl', [depthA]).familyMatch).toBe(true);
    expect(resolve(layer('depth'), 'sdxl', [normal])).toMatchObject({ familyMatch: false, how: 'only', model: normal });
    expect(resolve(layer('depth'), 'sdxl', [normal], { 'sdxl:depth': normal.key }).familyMatch).toBe(false);
    expect(resolve(layer('depth'), 'flux', [fluxUnion])).toMatchObject({ familyMatch: true, how: 'union' });
    expect(resolve(layer('depth', { model: pose.key }), 'sdxl', [pose])).toMatchObject({
      familyMatch: false,
      how: 'explicit',
    });
    expect(resolve(layer('depth'), 'sdxl', [depthA, depthB]).familyMatch).toBe(false);
  });
});

describe('isRememberableControlPick', () => {
  const unpurposed = model('my-sdxl-control', 'sdxl');
  const models = [depthA, depthB, normal, pose, sdxlUnion, unpurposed];

  it.each([
    ['depth_anything_depth_estimation', 'the depth detector'],
    ['normal_map', 'the normal-map detector'],
  ])('remembers only a model for the primary family or a union covering it, whatever %s (%s)', (detector) => {
    const resolution = resolve(layer('depth', { detector }), 'sdxl', models);

    expect(isRememberableControlPick(resolution, depthA)).toBe(true);
    expect(isRememberableControlPick(resolution, sdxlUnion)).toBe(true);
    expect(isRememberableControlPick(resolution, normal)).toBe(false);
    expect(isRememberableControlPick(resolution, pose)).toBe(false);
    expect(isRememberableControlPick(resolution, unpurposed)).toBe(false);
  });

  it('remembers nothing for an untyped layer or a model the base cannot run', () => {
    expect(isRememberableControlPick(resolve(layer(undefined), 'sdxl', models), depthA)).toBe(false);
    expect(isRememberableControlPick(resolve(layer('depth'), 'sdxl', models), sd1Depth)).toBe(false);
  });

  it('keeps a secondary-family pick from unsettling other layers of the type', () => {
    const remembered: ControlModelMemory = { 'sdxl:depth': depthB.key };
    const normalLayer = resolve(layer('depth', { detector: 'normal_map' }), 'sdxl', models, remembered);
    const memory = isRememberableControlPick(normalLayer, normal) ? { 'sdxl:depth': normal.key } : remembered;

    expect(
      resolve(layer('depth', { detector: 'depth_anything_depth_estimation' }), 'sdxl', models, memory)
    ).toMatchObject({ how: 'remembered', model: depthB, unresolved: null });
  });
});

describe('resolveControlModel InstantX modes', () => {
  const fluxUnion = model('FLUX.1-dev-Controlnet-Union-Pro', 'flux');

  it.each([
    ['edges', 0],
    ['tile', 1],
    ['depth', 2],
    ['pose', 4],
  ] as const)('gives a FLUX union %s layer mode %s', (type, mode) => {
    expect(resolve(layer(type), 'flux', [fluxUnion])).toMatchObject({ instantxControlMode: mode, model: fluxUnion });
  });

  it('leaves the mode out for families InstantX lacks, dedicated FLUX models and other bases', () => {
    expect(
      resolve(layer('edges', { detector: 'mlsd_detection', model: fluxUnion.key }), 'flux', [fluxUnion])
    ).toMatchObject({
      how: 'explicit',
      instantxControlMode: null,
    });
    expect(
      resolve(layer('depth'), 'flux', [model('flux-depth-lora', 'flux', 'control_lora')]).instantxControlMode
    ).toBeNull();
    expect(resolve(layer('depth'), 'z-image', [model('z-image-fun', 'z-image')]).instantxControlMode).toBeNull();
  });
});

describe('getTypeAvailability', () => {
  const byType = (availability: ReturnType<typeof getTypeAvailability>) =>
    Object.fromEntries(availability.map(({ count, state, type }) => [type, [state, count]]));

  it('hints how each type would resolve on a fresh layer', () => {
    expect(byType(getTypeAvailability('sdxl', [depthA, depthB, pose, sdxlUnion, sd1Depth], supportsAll))).toEqual({
      color: ['missing', 0],
      depth: ['ask', 2],
      edges: ['union', 1],
      lineart: ['union', 1],
      pose: ['ready', 1],
      sketch: ['union', 1],
      softedge: ['union', 1],
      tile: ['union', 1],
    });
    expect(
      byType(
        getTypeAvailability(
          'flux',
          [model('flux-depth-lora', 'flux', 'control_lora'), model('flux-union', 'flux')],
          supportsAll
        )
      )
    ).toMatchObject({ depth: ['ready', 1], edges: ['union', 1], lineart: ['missing', 0], tile: ['union', 1] });
  });
});
