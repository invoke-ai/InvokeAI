import type { ModelConfig } from '@features/models';
import type { CanvasControlLayerContract } from '@workbench/canvas-engine/api';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { CONTROL_ADAPTER_KINDS, CONTROL_VALIDATION_REASONS } from '@features/generation/graph';
import { createInstance } from 'i18next';
import { describe, expect, it } from 'vitest';

import {
  createControlLayerCheckContext,
  describeControlLayer,
  describeControlLayerIssue,
  describeControlLayerReason,
  hasControlLayerContent,
} from './controlLayerChecks';

const model = (key: string, base: string, type: string): ModelConfig => ({ base, key, name: key, type }) as ModelConfig;

type MainModel = { base: string; variant?: string };

/** The ambient warning the panel and the layer row show: per-generation limits left to Invoke. */
const ambientReason = (layer: CanvasControlLayerContract, mainModel: MainModel, models: readonly ModelConfig[]) =>
  describeControlLayer(layer, { ...createControlLayerCheckContext(mainModel.base), mainModel, models }).reason;

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

describe('describeControlLayer ambient warnings', () => {
  it('warns that a fresh layer needs a type, even without content', () => {
    const layer = controlLayer('fresh', { source: { bitmap: null, type: 'paint' } });

    expect(ambientReason(layer, sdxl, models)).toBe('missing_type');
  });

  it('returns null for a healthy layer and ignores per-kind limits', () => {
    expect(ambientReason(controlLayer('ok', { model: 'sdxl-control' }), sdxl, models)).toBeNull();
  });

  it('warns that a Klein model cannot run the FLUX.2 control, even on a fresh typed layer without a model', () => {
    const fresh = controlLayer('fresh', {
      derivation: { controlType: 'depth', detection: null },
      source: { bitmap: null, type: 'paint' },
    });
    const picked = controlLayer('picked', { kind: 'flux2_control', model: 'flux2-union', weight: 0.75 });
    const flux2Models = [model('flux2-union', 'flux2', 'controlnet')];

    expect(ambientReason(fresh, { base: 'flux2', variant: 'klein_4b' }, flux2Models)).toBe('flux2_klein_control');
    expect(ambientReason(picked, { base: 'flux2', variant: 'dev' }, flux2Models)).toBeNull();
  });

  it('warns that an Edit model cannot run the Qwen-Image control', () => {
    const qwenModels = [model('qwen-union', 'qwen-image', 'controlnet')];
    const layer = controlLayer('qwen', { kind: 'qwen_image_control', model: 'qwen-union', weight: 0.85 });

    expect(ambientReason(layer, { base: 'qwen-image' }, qwenModels)).toBeNull();
    expect(ambientReason(layer, { base: 'qwen-image', variant: 'edit' }, qwenModels)).toBe('qwen_image_edit_control');
  });
});

describe('describeControlLayer after earlier layers', () => {
  const fluxModels = [model('flux-lora', 'flux', 'control_lora'), model('sdxl-lora', 'sdxl', 'control_lora')];
  const target = controlLayer('target', { kind: 'control_lora', model: 'flux-lora' });

  it('lets only an earlier layer that passes claim the one Control LoRA slot, as the invocation does', () => {
    const judge = (earlier: CanvasControlLayerContract[]) =>
      describeControlLayer(target, {
        ...createControlLayerCheckContext('flux'),
        earlier,
        mainModel: { base: 'flux' },
        models: fluxModels,
      }).reason;

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

  it('still warns that an empty layer needs one of several fitting models', () => {
    expect(ambientReason(typed('empty', {}, { source: { bitmap: null, type: 'paint' } }), sdxl, depthModels)).toBe(
      'choose_model'
    );
  });

  it('warns of a map detection has not caught up with', () => {
    const pending = typed('pending', { detection: { settings: {}, type: 'depth_anything_depth_estimation' } });

    expect(ambientReason(pending, sdxl, [depthModels[0]!])).toBe('detection_pending');
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
      'widgets.layers.control.invalidLayer',
    ];
    expect(keys.filter((key) => !english.exists(key))).toEqual([]);
  });

  it('words a reason and names the layer, from the locale', () => {
    // The Adapter type control is gone: a layer the main model cannot run recovers by choosing a type.
    expect(describeControlLayerReason(t, 'switch_adapter_kind')).toBe(
      'Choose a control type so a model for the selected main model can be picked.'
    );
    expect(describeControlLayerIssue(t, { code: 'missing_model', layerName: 'Sketch' })).toBe(
      `Control layer "Sketch": ${t('widgets.layers.control.validation.missing_model')}`
    );
  });
});
