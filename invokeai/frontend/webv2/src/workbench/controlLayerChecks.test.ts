import type { ModelConfig } from '@features/models';
import type { CanvasControlLayerContract } from '@workbench/canvas-engine/api';

import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
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
