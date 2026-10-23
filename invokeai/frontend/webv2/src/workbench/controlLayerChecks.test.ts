import type { ModelConfig } from '@features/models';
import type { CanvasControlLayerContract } from '@workbench/canvas-engine/api';

import { resetArchitectureCapabilities } from '@features/generation/core/architectureCapabilities';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { CONTROL_ADAPTER_KINDS, CONTROL_VALIDATION_REASONS } from '@features/generation/graph';
import { createInstance } from 'i18next';
import { describe, expect, it } from 'vitest';

import {
  createControlLayerCheckContext,
  describeControlLayer,
  describeControlLayerIssue,
  describeControlLayerReason,
  getBlockingControlLayerIssues,
  getControlLayerAttentionReason,
  getControlLayerReasonInSequence,
  hasControlLayerContent,
} from './controlLayerChecks';

const model = (key: string, base: string, type: string): ModelConfig => ({ base, key, name: key, type }) as ModelConfig;

const controlLayer = (
  id: string,
  overrides: {
    isEnabled?: boolean;
    kind?: CanvasControlLayerContract['adapter']['kind'];
    model?: string | null;
    source?: CanvasControlLayerContract['source'];
    weight?: number;
    derivation?: Pick<CanvasControlLayerContract, 'controlType' | 'detection' | 'filter' | 'original'>;
  } = {}
): CanvasControlLayerContract => ({
  ...overrides.derivation,
  adapter: {
    beginEndStepPct: [0, 1],
    controlMode: null,
    kind: overrides.kind ?? 'controlnet',
    model: overrides.model ?? null,
    weight: overrides.weight ?? 1,
  },
  blendMode: 'normal',
  id,
  isEnabled: overrides.isEnabled ?? true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: overrides.source ?? { image: { height: 64, imageName: `${id}.png`, width: 64 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: true,
});

const sdxl = { base: 'sdxl' };
const models = [model('sdxl-control', 'sdxl', 'controlnet'), model('sd1-control', 'sd-1', 'controlnet')];

seedArchitectureCapabilities();

describe('hasControlLayerContent', () => {
  it('matches the pipeline content gate', () => {
    expect(hasControlLayerContent(controlLayer('image'))).toBe(true);
    expect(
      hasControlLayerContent(
        controlLayer('painted', { source: { bitmap: { height: 8, width: 8 } as never, type: 'paint' } })
      )
    ).toBe(true);
    expect(hasControlLayerContent(controlLayer('empty', { source: { bitmap: null, type: 'paint' } }))).toBe(false);
  });
});

describe('getBlockingControlLayerIssues', () => {
  it('leaves out a content-bearing layer with neither a type nor a model: it warns but contributes nothing', () => {
    const layer = controlLayer('Control Layer 1');

    expect(getBlockingControlLayerIssues({ layers: [layer], mainModel: sdxl, models })).toEqual([]);
    expect(getControlLayerAttentionReason(layer, sdxl, models)).toBe('missing_type');
  });

  it('blocks an untyped layer whose model is no longer installed', () => {
    expect(
      getBlockingControlLayerIssues({
        layers: [controlLayer('legacy', { model: 'uninstalled' })],
        mainModel: sdxl,
        models,
      })
    ).toMatchObject([{ code: 'missing_model', layerId: 'legacy' }]);
  });

  it('reports a typed content-bearing layer no model resolves, in structured form', () => {
    const issues = getBlockingControlLayerIssues({
      layers: [controlLayer('Control Layer 1', { derivation: { controlType: 'pose', detection: null } })],
      mainModel: sdxl,
      models,
    });

    expect(issues).toEqual([
      {
        code: 'no_model_for_type',
        layerId: 'Control Layer 1',
        layerName: 'Control Layer 1',
        suggestedKind: null,
      },
    ]);
  });

  it('skips disabled, content-less, and non-control layers', () => {
    const issues = getBlockingControlLayerIssues({
      layers: [
        controlLayer('disabled', { isEnabled: false }),
        controlLayer('empty', { source: { bitmap: null, type: 'paint' } }),
      ],
      mainModel: sdxl,
      models,
    });

    expect(issues).toEqual([]);
  });

  it('reports an incompatible base model', () => {
    const issues = getBlockingControlLayerIssues({
      layers: [controlLayer('mismatch', { model: 'sd1-control' })],
      mainModel: sdxl,
      models,
    });

    expect(issues).toMatchObject([{ code: 'incompatible_base' }]);
  });

  it('advances the control LoRA limit counter only past valid layers', () => {
    const fluxModels = [model('flux-lora', 'flux', 'control_lora')];
    const issues = getBlockingControlLayerIssues({
      layers: [
        controlLayer('first', { kind: 'control_lora', model: 'flux-lora' }),
        controlLayer('second', { kind: 'control_lora', model: 'flux-lora' }),
      ],
      mainModel: { base: 'flux' },
      models: fluxModels,
    });

    expect(issues).toEqual([
      {
        code: 'control_lora_limit',
        layerId: 'second',
        layerName: 'second',
        suggestedKind: null,
      },
    ]);
  });

  it('reports Anima LLLite layers the denoiser would refuse, and only those', () => {
    const animaModels = [
      { ...model('sketch', 'anima', 'controlnet'), cond_in_channels: 3 },
      { ...model('depth', 'anima', 'controlnet'), cond_in_channels: 3 },
      { ...model('inpaint', 'anima', 'controlnet'), cond_in_channels: 4 },
    ];
    const lllite = (id: string, key: string) => controlLayer(id, { kind: 'anima_lllite', model: key });

    const issues = getBlockingControlLayerIssues({
      layers: [
        lllite('sketch', 'sketch'),
        lllite('depth', 'depth'),
        lllite('again', 'sketch'),
        lllite('inpaint', 'inpaint'),
        controlLayer('saved-as-controlnet', { model: 'depth' }),
      ],
      mainModel: { base: 'anima' },
      models: animaModels,
    });

    expect(issues.map(({ code, layerId, suggestedKind }) => [layerId, code, suggestedKind])).toEqual([
      ['again', 'duplicate_lllite_model', null],
      ['inpaint', 'lllite_inpaint_adapter', null],
      ['saved-as-controlnet', 'switch_adapter_kind', 'anima_lllite'],
    ]);
  });

  it('lets one Qwen-Image control through, and none on an Edit model', () => {
    const qwenModels = [model('qwen-union', 'qwen-image', 'controlnet')];
    const layers = [
      controlLayer('first', { kind: 'qwen_image_control', model: 'qwen-union', weight: 0.85 }),
      controlLayer('second', { kind: 'qwen_image_control', model: 'qwen-union', weight: 0.85 }),
    ];
    const codes = (mainModel: { base: string; variant?: string }) =>
      getBlockingControlLayerIssues({ layers, mainModel, models: qwenModels }).map(({ code, layerId }) => [
        layerId,
        code,
      ]);

    expect(codes({ base: 'qwen-image' })).toEqual([['second', 'qwen_image_control_limit']]);
    expect(codes({ base: 'qwen-image', variant: 'edit' })).toEqual([
      ['first', 'qwen_image_edit_control'],
      ['second', 'qwen_image_edit_control'],
    ]);
  });

  it('lets one FLUX.2 control through on [dev], and none on Klein', () => {
    const flux2Models = [model('flux2-union', 'flux2', 'controlnet')];
    const layers = [
      controlLayer('first', { kind: 'flux2_control', model: 'flux2-union', weight: 0.75 }),
      controlLayer('second', { kind: 'flux2_control', model: 'flux2-union', weight: 0.75 }),
    ];
    const codes = (mainModel: { base: string; variant?: string }) =>
      getBlockingControlLayerIssues({ layers, mainModel, models: flux2Models }).map(({ code, layerId }) => [
        layerId,
        code,
      ]);

    expect(codes({ base: 'flux2', variant: 'dev' })).toEqual([['second', 'flux2_control_limit']]);
    expect(codes({ base: 'flux2', variant: 'klein_9b' })).toEqual([
      ['first', 'flux2_klein_control'],
      ['second', 'flux2_klein_control'],
    ]);
  });

  it('reports issues in document order', () => {
    const pose = { derivation: { controlType: 'pose', detection: null } } as const;
    const issues = getBlockingControlLayerIssues({
      layers: [controlLayer('top', pose), controlLayer('bottom', pose)],
      mainModel: sdxl,
      models,
    });

    expect(issues.map((issue) => issue.layerName)).toEqual(['top', 'bottom']);
  });
});

describe('getControlLayerAttentionReason', () => {
  it('warns that a fresh layer needs a type, even without content', () => {
    const layer = controlLayer('fresh', { source: { bitmap: null, type: 'paint' } });

    expect(getControlLayerAttentionReason(layer, sdxl, models)).toBe('missing_type');
  });

  it('returns null for a healthy layer and ignores per-kind limits', () => {
    expect(getControlLayerAttentionReason(controlLayer('ok', { model: 'sdxl-control' }), sdxl, models)).toBeNull();
  });

  it('warns that a Klein model cannot run the FLUX.2 control, even on a fresh typed layer without a model', () => {
    const fresh = controlLayer('fresh', {
      derivation: { controlType: 'depth', detection: null },
      source: { bitmap: null, type: 'paint' },
    });
    const picked = controlLayer('picked', { kind: 'flux2_control', model: 'flux2-union', weight: 0.75 });
    const flux2Models = [model('flux2-union', 'flux2', 'controlnet')];

    expect(getControlLayerAttentionReason(fresh, { base: 'flux2', variant: 'klein_4b' }, flux2Models)).toBe(
      'flux2_klein_control'
    );
    expect(getControlLayerAttentionReason(picked, { base: 'flux2', variant: 'dev' }, flux2Models)).toBeNull();
  });

  it('warns that an Edit model cannot run the Qwen-Image control', () => {
    const qwenModels = [model('qwen-union', 'qwen-image', 'controlnet')];
    const layer = controlLayer('qwen', { kind: 'qwen_image_control', model: 'qwen-union', weight: 0.85 });

    expect(getControlLayerAttentionReason(layer, { base: 'qwen-image' }, qwenModels)).toBeNull();
    expect(getControlLayerAttentionReason(layer, { base: 'qwen-image', variant: 'edit' }, qwenModels)).toBe(
      'qwen_image_edit_control'
    );
  });
});

describe('getControlLayerReasonInSequence', () => {
  const fluxModels = [model('flux-lora', 'flux', 'control_lora'), model('sdxl-lora', 'sdxl', 'control_lora')];
  const target = controlLayer('target', { kind: 'control_lora', model: 'flux-lora' });

  it('lets only an earlier layer that passes claim the one Control LoRA slot, as the invocation does', () => {
    const judge = (earlier: CanvasControlLayerContract[]) =>
      getControlLayerReasonInSequence({ earlier, layer: target, mainModel: { base: 'flux' }, models: fluxModels });

    expect(judge([])).toBeNull();
    expect(judge([controlLayer('invalid', { kind: 'control_lora', model: 'sdxl-lora' })])).toBeNull();
    expect(judge([controlLayer('valid', { kind: 'control_lora', model: 'flux-lora' })])).toBe('control_lora_limit');
  });
});

describe('typed control layers', () => {
  const depthModels = [model('sdxl-depth-a', 'sdxl', 'controlnet'), model('sdxl-depth-b', 'sdxl', 'controlnet')];
  const typed = (
    id: string,
    derivation: Partial<Pick<CanvasControlLayerContract, 'detection' | 'filter' | 'original'>> = {},
    overrides: Parameters<typeof controlLayer>[1] = {}
  ) => controlLayer(id, { ...overrides, derivation: { controlType: 'depth', detection: null, ...derivation } });
  const depthDetection = { settings: {}, type: 'depth_anything_depth_estimation' };

  it('asks for a model when several fit, and resolves with the remembered pick', () => {
    const layers = [typed('depth')];

    expect(getBlockingControlLayerIssues({ layers, mainModel: sdxl, models: depthModels })).toMatchObject([
      { code: 'choose_model' },
    ]);
    expect(
      getBlockingControlLayerIssues({
        context: createControlLayerCheckContext('sdxl', { 'sdxl:depth': 'sdxl-depth-b' }),
        layers,
        mainModel: sdxl,
        models: depthModels,
      })
    ).toEqual([]);
  });

  it('tells no control models apart from none for the type', () => {
    expect(getBlockingControlLayerIssues({ layers: [typed('depth')], mainModel: sdxl, models: [] })).toMatchObject([
      { code: 'no_control_models' },
    ]);
  });

  it('never blocks on an empty layer, though it still warns', () => {
    const empty = typed('empty', {}, { source: { bitmap: null, type: 'paint' } });

    expect(getBlockingControlLayerIssues({ layers: [empty], mainModel: sdxl, models: depthModels })).toEqual([]);
    expect(getControlLayerAttentionReason(empty, sdxl, depthModels)).toBe('choose_model');
  });

  it('blocks on a map detection has not caught up with, after the model reasons', () => {
    const pending = typed('pending', { detection: depthDetection });

    expect(
      getBlockingControlLayerIssues({ layers: [pending], mainModel: sdxl, models: [depthModels[0]!] })
    ).toMatchObject([{ code: 'detection_pending' }]);
    expect(getBlockingControlLayerIssues({ layers: [pending], mainModel: sdxl, models: depthModels })).toMatchObject([
      { code: 'choose_model' },
    ]);
    expect(getControlLayerAttentionReason(pending, sdxl, [depthModels[0]!])).toBe('detection_pending');
  });

  it('counts per-generation limits by the kind the resolved model runs, not the stored kind', () => {
    const fluxDepth = [model('flux-depth-lora', 'flux', 'control_lora')];
    const issues = getBlockingControlLayerIssues({
      layers: [typed('first'), typed('second')],
      mainModel: { base: 'flux' },
      models: fluxDepth,
    });

    expect(issues.map(({ code, layerId }) => [layerId, code])).toEqual([['second', 'control_lora_limit']]);
  });

  it('still resolves models before capabilities load, and reports that they have not', () => {
    const context = createControlLayerCheckContext('flux');
    const judge = () =>
      getBlockingControlLayerIssues({
        context,
        layers: [typed('depth')],
        mainModel: { base: 'flux' },
        models: [model('flux-depth-t2i', 'flux', 't2i_adapter')],
      });

    // Loaded, FLUX runs no T2I-Adapter, so no model fits.
    expect(judge()).toMatchObject([{ code: 'no_control_models' }]);
    // The context reads capabilities when it judges, not when it is made.
    resetArchitectureCapabilities();
    expect(judge()).toMatchObject([{ code: 'capabilities_unavailable' }]);
  });
});

describe('describeControlLayer', () => {
  const sdxlDepth = [model('sdxl-depth', 'sdxl', 'controlnet')];
  const ctx = { ...createControlLayerCheckContext('sdxl'), mainModel: sdxl, models: sdxlDepth };
  const detection = { settings: {}, type: 'depth_anything_depth_estimation' };
  const original = { image: { height: 64, imageName: 'photo.png', width: 64 }, type: 'image' } as const;
  const depth = (derivation: Pick<CanvasControlLayerContract, 'detection' | 'filter' | 'original'>, empty = false) =>
    controlLayer('depth', {
      derivation: { controlType: 'depth', ...derivation },
      ...(empty ? { source: { bitmap: null, type: 'paint' } } : {}),
    });

  it.each([
    ['empty', depth({ detection }, true)],
    ['none', controlLayer('untyped', { model: 'sdxl-depth' })],
    ['none', depth({ detection: null })],
    ['stale', depth({ detection })],
    ['current', depth({ detection, filter: detection, original })],
    ['no-input', depth({ detection, filter: { settings: {}, type: 'canny_edge_detection' } })],
  ] as const)('reports a %s map', (map, layer) => {
    expect(describeControlLayer(layer, ctx).map).toBe(map);
  });

  it('resolves across every base without a main model, reporting only resolution and detection', () => {
    const noMain = {
      ...createControlLayerCheckContext(null),
      mainModel: null,
      models: [model('sd1-depth', 'sd-1', 'controlnet'), model('flux-t2i-depth', 'flux', 't2i_adapter')],
    };

    expect(describeControlLayer(depth({ detection: null }), noMain)).toMatchObject({
      reason: 'choose_model',
      resolution: { candidates: noMain.models, how: 'ask' },
    });
    expect(describeControlLayer(depth({ detection }), { ...noMain, models: [noMain.models[0]!] })).toMatchObject({
      map: 'stale',
      reason: 'detection_pending',
      resolution: { how: 'only' },
    });
  });

  it('carries the resolution, and with earlier layers judges per-generation limits as Invoke does', () => {
    const flux = {
      ...createControlLayerCheckContext('flux'),
      mainModel: { base: 'flux' },
      models: [model('flux-depth-lora', 'flux', 'control_lora')],
    };
    const layer = depth({ detection: null });

    expect(describeControlLayer(layer, flux)).toMatchObject({
      reason: null,
      resolution: { how: 'only', kind: 'control_lora' },
    });
    expect(describeControlLayer(layer, { ...flux, earlier: [depth({ detection: null })] }).reason).toBe(
      'control_lora_limit'
    );
  });
});

const englishCatalogModules = import.meta.glob('../../public/locales/en.json', { eager: true, import: 'default' });
const english = createInstance();
await english.init({
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: Object.values(englishCatalogModules)[0] as Record<string, unknown> } },
});
const t = english.t.bind(english) as (key: string, options?: Record<string, unknown>) => string;

describe('control layer wording', () => {
  it('has an English string for every reason code, adapter kind and hidden-model note', () => {
    const keys = [
      ...[...CONTROL_VALIDATION_REASONS, 'detection_pending'].map(
        (code) => `widgets.layers.control.validation.${code}`
      ),
      ...CONTROL_ADAPTER_KINDS.map((kind) => `widgets.layers.control.kinds.${kind}`),
      'widgets.layers.control.hiddenModels.lllite_inpaint_adapter',
      'widgets.layers.control.hiddenModels.lllite_channels_unknown',
      'widgets.layers.control.switchKind',
      'widgets.layers.control.invalidLayer',
    ];
    expect(keys.filter((key) => !english.exists(key))).toEqual([]);
  });

  it('names the kind to switch to and the layer, from the locale', () => {
    expect(describeControlLayerReason(t, 'switch_adapter_kind', 'anima_lllite')).toContain(
      t('widgets.layers.control.kinds.anima_lllite')
    );
    expect(describeControlLayerIssue(t, { code: 'missing_model', layerName: 'Sketch', suggestedKind: null })).toBe(
      `Control layer "Sketch": ${t('widgets.layers.control.validation.missing_model')}`
    );
  });
});
