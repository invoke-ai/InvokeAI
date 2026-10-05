import type { ModelConfig } from '@features/models';
import type { CanvasControlLayerContract } from '@workbench/canvas-engine/api';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { CONTROL_ADAPTER_KINDS, CONTROL_VALIDATION_REASONS } from '@features/generation/graph';
import { createInstance } from 'i18next';
import { describe, expect, it } from 'vitest';

import {
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
  } = {}
): CanvasControlLayerContract => ({
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
  it('reports a content-bearing model-less control layer with a human sentence', () => {
    const issues = getBlockingControlLayerIssues({
      layers: [controlLayer('Control Layer 1')],
      mainModel: sdxl,
      models,
    });

    expect(issues).toEqual([
      {
        code: 'missing_model',
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

  it('reports issues in document order', () => {
    const issues = getBlockingControlLayerIssues({
      layers: [controlLayer('top'), controlLayer('bottom')],
      mainModel: sdxl,
      models,
    });

    expect(issues.map((issue) => issue.layerName)).toEqual(['top', 'bottom']);
  });
});

describe('getControlLayerAttentionReason', () => {
  it('warns about a missing model even without content', () => {
    const layer = controlLayer('fresh', { source: { bitmap: null, type: 'paint' } });

    expect(getControlLayerAttentionReason(layer, 'sdxl', models)).toBe('missing_model');
  });

  it('returns null for a healthy layer and ignores per-kind limits', () => {
    expect(getControlLayerAttentionReason(controlLayer('ok', { model: 'sdxl-control' }), 'sdxl', models)).toBeNull();
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
      ...CONTROL_VALIDATION_REASONS.map((code) => `widgets.layers.control.validation.${code}`),
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
