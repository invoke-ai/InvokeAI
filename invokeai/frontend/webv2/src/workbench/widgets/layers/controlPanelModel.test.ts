import type { ModelConfig } from '@features/models';
import type { CanvasControlLayerContract, ControlType } from '@workbench/canvas-engine/api';
import type { ControlDetectionStatus } from '@workbench/canvas-operations/api';
import type { ControlImageState } from '@workbench/canvasLayerContent';
import type { ControlMapState } from '@workbench/controlLayerChecks';

import { createControlLayer } from '@workbench/canvas-engine/api';
import { getControlImageState } from '@workbench/canvasLayerContent';
import { getTypeAvailability, resolveControlModel, type ControlModelMemory } from '@workbench/controlModelResolution';
import { createInstance } from 'i18next';
import { describe, expect, it } from 'vitest';

import {
  deriveControlPanel,
  formatPanelText,
  type ControlPanelInput,
  type ControlPanelModel,
  type PanelText,
} from './controlPanelModel';

const model = (key: string, base: string, type = 'controlnet', extra: Partial<ModelConfig> = {}): ModelConfig =>
  ({ base, key, name: key, source: `hf/${key}`, type, ...extra }) as ModelConfig;

const depthA = model('sdxl-depth-a', 'sdxl');
const depthB = model('sdxl-depth-b', 'sdxl');
const pose = model('sdxl-openpose', 'sdxl');
const t2iCanny = model('t2i-adapter-canny-sdxl', 'sdxl', 't2i_adapter');
const fluxUnion = model('flux-controlnet-union-pro', 'flux');

const IMAGE = { image: { height: 64, imageName: 'photo.png', width: 96 }, type: 'image' } as const;
const IDLE: ControlDetectionStatus = { status: 'idle' };

const controlLayer = (
  overrides: Partial<CanvasControlLayerContract> & { model?: string | null } = {}
): CanvasControlLayerContract => {
  const { model: adapterModel, ...rest } = overrides;
  const created = createControlLayer({ base: 'sdxl', id: 'control-1', name: 'Control Layer 1' });
  const adapter = { ...created.adapter, ...rest.adapter };
  return { ...created, ...rest, adapter: adapterModel === undefined ? adapter : { ...adapter, model: adapterModel } };
};

/** A typed layer holding a photo it detects `detector` from, with the map current. */
const detected = (controlType: ControlType, detector: string | null, extra: Partial<CanvasControlLayerContract> = {}) =>
  controlLayer({
    controlType,
    detection: detector ? { settings: {}, type: detector } : null,
    ...(detector ? { filter: { settings: {}, type: detector }, original: IMAGE } : {}),
    source: IMAGE,
    ...extra,
  });

const derive = (
  layer: CanvasControlLayerContract,
  {
    base = 'sdxl',
    imageState,
    map = 'current',
    models = [depthA, pose],
    reason = null,
    remembered = {},
    status = IDLE,
  }: {
    base?: string | null;
    imageState?: ControlImageState;
    map?: ControlMapState;
    models?: readonly ModelConfig[];
    reason?: ControlPanelInput['reason'];
    remembered?: ControlModelMemory;
    status?: ControlDetectionStatus;
  } = {}
): ControlPanelModel => {
  const supportsKind = () => true;
  return deriveControlPanel({
    availability: getTypeAvailability(base, models, supportsKind),
    base: base ? { id: base, label: base.toUpperCase() } : null,
    imageState: imageState ?? getControlImageState(layer, false),
    layer,
    map,
    reason,
    resolution: resolveControlModel({ base, layer, models, remembered, supportsKind }),
    status,
  });
};

/** Every text a model carries, for the catalog check at the end. */
const seen: PanelText[] = [];
const collect = (panel: ControlPanelModel): ControlPanelModel => {
  const texts = [
    ...Object.values(panel.type.hints).map((hint) => hint.text),
    panel.detection.valueText,
    panel.detection.settingsUnavailable,
    ...panel.detection.options.flatMap((option) => [option.label, option.hint]),
    panel.model.valueText,
    panel.model.caption,
    panel.adapter.stepsReadout,
  ];
  seen.push(...texts.filter((text): text is PanelText => text !== null));
  return panel;
};
const panelOf = (...args: Parameters<typeof derive>) => collect(derive(...args));

describe('control panel design states', () => {
  it('1: a new empty layer asks for nothing yet and waits for a type', () => {
    const panel = panelOf(controlLayer());

    expect(panel.image).toEqual({ imageName: null, state: 'empty' });
    expect(panel.type).toMatchObject({ invalid: false, value: null });
    expect(panel.detection).toMatchObject({
      disabled: true,
      options: [],
      valueText: { key: 'detectionSetByType' },
    });
    expect(panel.model).toMatchObject({
      action: null,
      caption: { key: 'typeMissing' },
      captionTone: 'muted',
      valueText: { key: 'modelValue.setByType' },
    });
  });

  it('2: an image without a type warns on Type and says the model follows it', () => {
    const panel = panelOf(controlLayer({ source: IMAGE }));

    expect(panel.image).toEqual({ imageName: 'photo.png', state: 'image' });
    expect(panel.type.invalid).toBe(true);
    expect(panel.model).toMatchObject({ caption: { key: 'modelCaption.chooseType' }, captionTone: 'warning' });
  });

  it('3: a depth layer with one depth model runs its default detector and names the only model', () => {
    const panel = panelOf(detected('depth', 'depth_anything_depth_estimation'));

    expect(panel.detection).toMatchObject({
      action: null,
      disabled: false,
      settings: { detector: 'depth_anything_depth_estimation', values: {} },
      settingsUnavailable: null,
      value: 'depth_anything_depth_estimation',
      valueText: { key: 'filters.depth_anything_depth_estimation' },
    });
    expect(panel.detection.options.map((option) => [option.value, option.disabled])).toEqual([
      ['depth_anything_depth_estimation', false],
      ['normal_map', false],
      ['none', false],
    ]);
    expect(panel.model).toMatchObject({
      action: null,
      caption: { key: 'modelCaption.only', refs: { kind: 'kinds.controlnet', type: 'typeNouns.depth' } },
      name: 'sdxl-depth-a',
      tone: 'default',
    });
  });

  it('4: several depth models ask for a choice and list them', () => {
    const panel = panelOf(detected('depth', 'depth_anything_depth_estimation'), { models: [depthA, depthB] });

    expect(panel.model).toMatchObject({
      caption: {
        key: 'modelCaption.ambiguous',
        lists: { models: ['sdxl-depth-a', 'sdxl-depth-b'] },
        values: { count: 2 },
      },
      name: null,
      tone: 'warning',
      valueText: { key: 'modelValue.choose', refs: { type: 'typeNouns.depth' } },
    });
  });

  it('5: an explicit or remembered pick says so and offers Reset', () => {
    const explicit = panelOf(
      detected('depth', 'depth_anything_depth_estimation', {
        adapter: { ...controlLayer().adapter, model: 'sdxl-depth-b' },
      }),
      { models: [depthA, depthB] }
    );
    const remembered = panelOf(detected('depth', 'depth_anything_depth_estimation'), {
      models: [depthA, depthB],
      remembered: { 'sdxl:depth': 'sdxl-depth-b' },
    });

    for (const panel of [explicit, remembered]) {
      expect(panel.model).toMatchObject({
        action: 'reset',
        caption: { key: 'modelCaption.pick' },
        name: 'sdxl-depth-b',
      });
    }
  });

  it('6: a union model covering the type is used and named as the union', () => {
    const panel = panelOf(detected('depth', 'depth_anything_depth_estimation'), { base: 'flux', models: [fluxUnion] });

    expect(panel.model).toMatchObject({ caption: { key: 'modelCaption.union' }, name: 'flux-controlnet-union-pro' });
  });

  it('7: nothing installed for the type points to finding one', () => {
    const panel = panelOf(detected('lineart', 'lineart_edge_detection'), { models: [depthA] });

    expect(panel.model).toMatchObject({
      action: 'find-one',
      caption: { key: 'modelCaption.noneForBase', refs: { type: 'typeNouns.lineart' }, values: { base: 'SDXL' } },
      tone: 'warning',
      valueText: { key: 'modelValue.noneForBase', values: { base: 'SDXL' } },
    });
    expect(panel.type.hints.lineart).toEqual({ text: { key: 'typeHint.notInstalled' }, tone: 'warning' });
  });

  it('8: no control models at all points to the starter models', () => {
    const panel = panelOf(detected('depth', 'depth_anything_depth_estimation'), { models: [] });

    expect(panel.model).toMatchObject({
      action: 'install-starters',
      caption: { key: 'modelCaption.noModels' },
      valueText: { key: 'modelValue.noModels' },
    });
  });
});

describe('control panel detection', () => {
  it('offers Tile only its None entry, with no settings', () => {
    const panel = panelOf(detected('tile', null));

    expect(panel.detection.options.map((option) => option.value)).toEqual(['none']);
    expect(panel.detection).toMatchObject({
      settings: null,
      settingsUnavailable: { key: 'detectionNoSettings', refs: { detector: 'detectionNone.tile' } },
      value: 'none',
      valueText: { key: 'detectionNone.tile' },
    });
  });

  it('shows the running detector, queued or processing', () => {
    const layer = detected('depth', 'depth_anything_depth_estimation');

    expect(panelOf(layer, { status: { status: 'queued' } }).detection.busy).toEqual({
      detector: 'depth_anything_depth_estimation',
    });
    expect(panelOf(layer, { status: { detector: 'normal_map', status: 'processing' } }).detection.busy).toEqual({
      detector: 'normal_map',
    });
    expect(panelOf(layer).detection.busy).toBeNull();
  });

  it('offers Run again after a failure, and Detect again for an out-of-date map at rest', () => {
    const layer = detected('depth', 'depth_anything_depth_estimation');
    const failed = panelOf(layer, { map: 'stale', status: { error: { code: 'process-failed' }, status: 'error' } });

    expect(failed.detection).toMatchObject({ action: 'retry', failed: { code: 'process-failed' }, tone: 'error' });
    expect(panelOf(layer, { map: 'stale' }).detection.action).toBe('detect');
    // A queued run is already on its way.
    expect(panelOf(layer, { map: 'stale', status: { status: 'queued' } }).detection.action).toBeNull();
  });

  it('keeps detectors out of reach until there is something to detect from', () => {
    const empty = panelOf(controlLayer({ controlType: 'depth', detection: null }));
    const mapOnly = panelOf(
      controlLayer({
        controlType: 'depth',
        detection: { settings: {}, type: 'depth_anything_depth_estimation' },
        filter: { settings: {}, type: 'depth_anything_depth_estimation' },
        source: IMAGE,
      }),
      { map: 'no-input' }
    );

    expect(empty.detection.options.map((option) => [option.value, option.disabled, option.hint?.key ?? null])).toEqual([
      ['depth_anything_depth_estimation', true, 'detectionNeedsImage'],
      ['normal_map', true, 'detectionNeedsImage'],
      ['none', false, null],
    ]);
    expect(mapOnly.detection.options[0]).toMatchObject({ disabled: true, hint: { key: 'detectionNoInput' } });
    expect(mapOnly.detection.settings).toBeNull();
    expect(mapOnly.detection.settingsUnavailable).toEqual({ key: 'detectionNoInput' });
    // An emptied layer that still names its detector says the image is what is missing.
    const emptied = panelOf(
      controlLayer({ controlType: 'depth', detection: { settings: {}, type: 'depth_anything_depth_estimation' } })
    );
    expect(emptied.detection.settingsUnavailable).toEqual({ key: 'detectionNeedsImage' });
  });
});

describe('control panel model caption', () => {
  it('says what a model is made for when it is not made for what the detector produces', () => {
    const normalModel = model('sdxl-normal', 'sdxl');
    const panel = panelOf(
      detected('depth', 'depth_anything_depth_estimation', {
        adapter: { ...controlLayer().adapter, model: 'sdxl-normal' },
      }),
      { models: [depthA, normalModel] }
    );

    expect(panel.model).toMatchObject({
      caption: { key: 'modelCaption.mismatch', refs: { family: 'families.depth', madeFor: 'families.normal' } },
      captionTone: 'warning',
    });
  });

  it('puts a problem with the model itself ahead of the explanation', () => {
    const panel = panelOf(detected('depth', 'depth_anything_depth_estimation'), { reason: 'control_lora_limit' });

    expect(panel.model).toMatchObject({ caption: { key: 'validation.control_lora_limit' }, captionTone: 'warning' });
  });

  it('keeps a working untyped layer quiet and names its kind', () => {
    const panel = panelOf(controlLayer({ model: 'sdxl-openpose', source: IMAGE }));

    expect(panel.type.invalid).toBe(false);
    expect(panel.model).toMatchObject({ caption: { key: 'kinds.controlnet' }, name: 'sdxl-openpose' });
  });
});

describe('control panel adapter rows', () => {
  it('dims Priority for a non-ControlNet model and reads legacy unbalanced as none checked', () => {
    const t2i = panelOf(detected('edges', 'canny_edge_detection'), { models: [t2iCanny] });
    const legacy = panelOf(
      detected('depth', 'depth_anything_depth_estimation', {
        adapter: { ...controlLayer().adapter, beginEndStepPct: [0.1, 0.75], controlMode: 'unbalanced' },
      })
    );

    expect(t2i.adapter).toMatchObject({ kind: 't2i_adapter', priority: { disabled: true } });
    expect(legacy.adapter).toEqual({
      kind: 'controlnet',
      priority: { disabled: false, value: null },
      stepsReadout: { key: 'stepsReadout', values: { begin: 10, end: 75 } },
    });
  });
});

describe('control panel type hints', () => {
  it('counts models, names union coverage and flags what is missing', () => {
    const { hints } = panelOf(controlLayer(), { models: [depthA, depthB, pose] }).type;

    expect(hints.depth).toEqual({ text: { key: 'typeHint.models', values: { count: 2 } }, tone: 'muted' });
    expect(hints.pose).toEqual({ text: { key: 'typeHint.models', values: { count: 1 } }, tone: 'muted' });
    expect(panelOf(controlLayer(), { base: 'flux', models: [fluxUnion] }).type.hints.edges).toEqual({
      text: { key: 'typeHint.union' },
      tone: 'muted',
    });
  });
});

const englishCatalog = import.meta.glob('../../../../public/locales/en.json', { eager: true, import: 'default' });
const english = createInstance();
await english.init({
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: Object.values(englishCatalog)[0] as Record<string, unknown> } },
});

describe('panel texts', () => {
  it('finds every key the states above produced in the English catalog', () => {
    const keys = seen.flatMap((text) => [text.key, ...Object.values(text.refs ?? {})]);
    const missing = keys
      .map((key) => `widgets.layers.control.${key}`)
      .filter((key) => !english.exists(key) && !english.exists(`${key}_other`));

    expect(seen.length).toBeGreaterThan(20);
    expect(missing).toEqual([]);
  });

  it('joins listed names as alternatives', () => {
    const text = { key: 'modelCaption.ambiguous', lists: { models: ['A', 'B', 'C'] }, values: { count: 3 } };

    expect(formatPanelText(english.t.bind(english), text, 'en')).toBe('3 models fit: A, B, or C');
  });
});
