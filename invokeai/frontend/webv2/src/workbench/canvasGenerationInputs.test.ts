import type { GenerateModelConfig, MainModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type {
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasRegionalGuidanceLayerContract,
  RegionalGuidanceReferenceImage,
  RegionalGuidanceReferenceImageAsset,
} from '@workbench/canvas-engine/contracts';

import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { createInstance } from 'i18next';
import { describe, expect, it } from 'vitest';

import {
  createCanvasInputJudge,
  describeCanvasInputIssue,
  getCanvasInputIssues,
  type CanvasInputJudgeParams,
} from './canvasGenerationInputs';

const model = (key: string, base: string, type: string): ModelConfig => ({ base, key, name: key, type }) as ModelConfig;
const main = (base: string, variant?: string): MainModelConfig => ({
  base,
  key: `${base}-main`,
  name: `${base} main`,
  type: 'main',
  ...(variant ? { variant } : {}),
});
const hostedModel: GenerateModelConfig = {
  base: 'external',
  capabilities: { modes: ['txt2img'] },
  format: 'external_api',
  key: 'hosted',
  name: 'OpenAI Image',
  provider_id: 'openai',
  type: 'external_image_generator',
};

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

const region = (
  id: string,
  overrides: Partial<CanvasRegionalGuidanceLayerContract> = {}
): CanvasRegionalGuidanceLayerContract => ({
  autoNegative: false,
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: { height: 64, imageName: `${id}.png`, width: 64 }, fill: { color: '#00ff00', style: 'solid' } },
  name: id,
  negativePrompt: null,
  opacity: 1,
  positivePrompt: 'a red hat',
  referenceImages: [],
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'regional_guidance',
  ...overrides,
});

const judgeParams = (
  mainModel: GenerateModelConfig,
  models: readonly ModelConfig[] | undefined,
  controlModelMemory: Record<string, string> = {}
): CanvasInputJudgeParams => ({ cfgScale: 7, controlModelMemory, model: mainModel, models });

const readinessIssues = (
  layers: CanvasLayerContract[],
  params: CanvasInputJudgeParams,
  options: { prompt?: string; rasterInFrame?: boolean } = {}
) =>
  getCanvasInputIssues({
    ...params,
    layers,
    prompt: options.prompt ?? 'a prompt',
    rasterInFrame: options.rasterInFrame ?? false,
    regionalLayersInFrame: layers.filter(
      (layer): layer is CanvasRegionalGuidanceLayerContract => layer.type === 'regional_guidance'
    ),
  });

const invocationVerdicts = (layers: CanvasLayerContract[], params: CanvasInputJudgeParams) => {
  const judge = createCanvasInputJudge(params);
  return layers.map((layer) =>
    layer.type === 'control' ? judge.control(layer) : judge.regional(layer as CanvasRegionalGuidanceLayerContract)
  );
};

const englishCatalogModules = import.meta.glob('../../public/locales/en.json', { eager: true, import: 'default' });
const english = createInstance();
await english.init({
  initAsync: false,
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: Object.values(englishCatalogModules)[0] as Record<string, unknown> } },
});
const t = english.t.bind(english) as (key: string, options?: Record<string, unknown>) => string;

seedArchitectureCapabilities();

describe('readiness and the invocation judge layers alike', () => {
  it('refuse every compositable control and regional layer of a hosted model, by name and in layer order', () => {
    const layers = [controlLayer('Canny guide'), region('Hat region'), controlLayer('off', { isEnabled: false })];
    const refusals = [
      {
        kind: 'hosted_refusal',
        modelName: 'OpenAI Image',
        refusal: { kind: 'control_layer', layerName: 'Canny guide' },
      },
      {
        kind: 'hosted_refusal',
        modelName: 'OpenAI Image',
        refusal: { kind: 'regional_guidance', layerName: 'Hat region' },
      },
    ];

    for (const models of [undefined, []]) {
      const params = judgeParams(hostedModel, models);
      expect(readinessIssues(layers, params)).toEqual(refusals);
      expect(invocationVerdicts(layers.slice(0, 2), params)).toEqual(
        refusals.map((issue) => ({ issue, status: 'blocked' }))
      );
    }
    expect(
      readinessIssues(layers, judgeParams(hostedModel, [])).map((issue) => describeCanvasInputIssue(t, issue))
    ).toEqual([
      'OpenAI Image does not support control layer "Canny guide". Disable it to generate.',
      'OpenAI Image does not support regional guidance layer "Hat region". Disable it to generate.',
    ]);
  });

  it('let the first Control LoRA through and block the second, counting only layers that passed', () => {
    const lora = model('flux-lora', 'flux', 'control_lora');
    const layers = [
      controlLayer('invalid', { kind: 'control_lora', model: 'sdxl-lora' }),
      controlLayer('first', { kind: 'control_lora', model: 'flux-lora' }),
      controlLayer('second', { kind: 'control_lora', model: 'flux-lora' }),
    ];
    const params = judgeParams(main('flux'), [lora, model('sdxl-lora', 'sdxl', 'control_lora')]);
    const blocked = { code: 'control_lora_limit', kind: 'control_layer', layerId: 'second', layerName: 'second' };

    expect(readinessIssues(layers, params)).toEqual([
      { code: 'incompatible_base', kind: 'control_layer', layerId: 'invalid', layerName: 'invalid' },
      blocked,
    ]);
    expect(invocationVerdicts(layers, params).map((verdict) => verdict.status)).toEqual([
      'blocked',
      'contributes',
      'blocked',
    ]);
    expect(invocationVerdicts(layers, params)[2]).toEqual({ issue: blocked, status: 'blocked' });
  });

  it('leave out a region a layout base cannot place, without blocking', () => {
    const layers = [region('uncaptioned', { positivePrompt: null }), region('sign', { positivePrompt: 'a shop sign' })];
    const params = judgeParams(main('moss-image'), []);

    expect(readinessIssues(layers, params)).toEqual([]);
    expect(invocationVerdicts(layers, params)).toEqual([
      { status: 'skipped' },
      {
        input: {
          autoNegative: false,
          id: 'sign',
          negativePrompt: null,
          positivePrompt: 'a shop sign',
          referenceImages: [],
        },
        status: 'contributes',
      },
    ]);
    expect(readinessIssues(layers, params, { prompt: ' ' })).toEqual([
      {
        kind: 'layout_prompt',
        message:
          'MOSS-Image needs a prompt to place regional guidance in its layout. Enter a prompt or disable the regional guidance layers.',
      },
    ]);
  });

  it('differ only while models load: readiness waits, the invocation refuses', () => {
    const layers = [controlLayer('sketch', { derivation: { controlType: 'depth', detection: null } })];
    const params = judgeParams(main('sdxl'), undefined);

    expect(readinessIssues(layers, params)).toEqual([]);
    expect(invocationVerdicts(layers, params)).toEqual([
      {
        issue: { code: 'models_unavailable', kind: 'control_layer', layerId: 'sketch', layerName: 'sketch' },
        status: 'awaiting_models',
      },
    ]);
  });
});

describe('getCanvasInputIssues', () => {
  const sdxl = main('sdxl');
  const models = [model('sdxl-control', 'sdxl', 'controlnet')];
  const controlIssues = (
    layers: CanvasLayerContract[],
    mainModel: GenerateModelConfig,
    installed: readonly ModelConfig[],
    controlModelMemory?: Record<string, string>
  ) => readinessIssues(layers, judgeParams(mainModel, installed, controlModelMemory));
  const codes = (issues: ReturnType<typeof controlIssues>) =>
    issues.map((issue) => (issue.kind === 'control_layer' ? [issue.layerId, issue.code] : issue));

  it('names raster content in the frame first for a text-only hosted model', () => {
    const editing: GenerateModelConfig = { ...hostedModel, capabilities: { modes: ['txt2img', 'img2img'] } };

    expect(readinessIssues([controlLayer('guide')], judgeParams(hostedModel, []), { rasterInFrame: true })).toEqual([
      { kind: 'hosted_refusal', modelName: 'OpenAI Image', refusal: { kind: 'raster_content' } },
      { kind: 'hosted_refusal', modelName: 'OpenAI Image', refusal: { kind: 'control_layer', layerName: 'guide' } },
    ]);
    expect(readinessIssues([], judgeParams(editing, []), { rasterInFrame: true })).toEqual([]);
  });

  it('blocks an untyped layer whose model is no longer installed', () => {
    expect(codes(controlIssues([controlLayer('legacy', { model: 'uninstalled' })], sdxl, models))).toEqual([
      ['legacy', 'missing_model'],
    ]);
  });

  it('reports a typed content-bearing layer no model resolves', () => {
    const pose = controlLayer('pose', { derivation: { controlType: 'pose', detection: null } });
    expect(codes(controlIssues([pose], sdxl, models))).toEqual([['pose', 'no_model_for_type']]);
  });

  it('skips disabled, content-less, and non-control layers', () => {
    expect(
      controlIssues(
        [
          controlLayer('disabled', { isEnabled: false, model: 'uninstalled' }),
          controlLayer('empty', { model: 'uninstalled', source: { bitmap: null, type: 'paint' } }),
          region('masked', { isEnabled: false }),
        ],
        sdxl,
        models
      )
    ).toEqual([]);
  });

  it('reports Anima LLLite layers the denoiser would refuse, and only those', () => {
    const animaModels = [
      { ...model('sketch', 'anima', 'controlnet'), cond_in_channels: 3 },
      { ...model('depth', 'anima', 'controlnet'), cond_in_channels: 3 },
      { ...model('inpaint', 'anima', 'controlnet'), cond_in_channels: 4 },
    ];
    const lllite = (id: string, key: string) => controlLayer(id, { kind: 'anima_lllite', model: key });

    expect(
      codes(
        controlIssues(
          [
            lllite('sketch', 'sketch'),
            lllite('depth', 'depth'),
            lllite('again', 'sketch'),
            lllite('inpaint', 'inpaint'),
            controlLayer('saved-as-controlnet', { model: 'depth' }),
          ],
          main('anima'),
          animaModels
        )
      )
    ).toEqual([
      ['again', 'duplicate_lllite_model'],
      ['inpaint', 'lllite_inpaint_adapter'],
      ['saved-as-controlnet', 'switch_adapter_kind'],
    ]);
  });

  it.each([
    [
      'Qwen-Image',
      'qwen-image',
      'qwen_image_control',
      0.85,
      'qwen_image_control_limit',
      'edit',
      'qwen_image_edit_control',
    ],
    ['FLUX.2', 'flux2', 'flux2_control', 0.75, 'flux2_control_limit', 'klein_9b', 'flux2_klein_control'],
  ] as const)('lets one %s control through, and none on a model that cannot run it', (...args) => {
    const [, base, kind, weight, limitCode, refusingVariant, refusalCode] = args;
    const union = [model(`${base}-union`, base, 'controlnet')];
    const layers = ['first', 'second'].map((id) => controlLayer(id, { kind, model: `${base}-union`, weight }));

    expect(codes(controlIssues(layers, main(base, base === 'flux2' ? 'dev' : undefined), union))).toEqual([
      ['second', limitCode],
    ]);
    expect(codes(controlIssues(layers, main(base, refusingVariant), union))).toEqual([
      ['first', refusalCode],
      ['second', refusalCode],
    ]);
  });

  describe('typed control layers', () => {
    const depthModels = [model('sdxl-depth-a', 'sdxl', 'controlnet'), model('sdxl-depth-b', 'sdxl', 'controlnet')];
    const typed = (id: string, detection: CanvasControlLayerContract['detection'] = null) =>
      controlLayer(id, { derivation: { controlType: 'depth', detection } });

    it('asks for a model when several fit, and resolves with the remembered pick', () => {
      expect(codes(controlIssues([typed('depth')], sdxl, depthModels))).toEqual([['depth', 'choose_model']]);
      expect(controlIssues([typed('depth')], sdxl, depthModels, { 'sdxl:depth': 'sdxl-depth-b' })).toEqual([]);
    });

    it('blocks on a map detection has not caught up with, after the model reasons', () => {
      const pending = typed('pending', { settings: {}, type: 'depth_anything_depth_estimation' });

      expect(codes(controlIssues([pending], sdxl, [depthModels[0]!]))).toEqual([['pending', 'detection_pending']]);
      expect(codes(controlIssues([pending], sdxl, depthModels))).toEqual([['pending', 'choose_model']]);
    });
  });
});

describe('createCanvasInputJudge', () => {
  it('hands a contributing control layer its adapter values and the model it resolves to', () => {
    const depth = { ...model('sdxl-depth', 'sdxl', 'controlnet'), hash: 'blake3:depth' };
    const layer = controlLayer('depth', { derivation: { controlType: 'depth', detection: null }, weight: 0.6 });

    expect(createCanvasInputJudge(judgeParams(main('sdxl'), [depth])).control(layer)).toEqual({
      input: {
        beginEndStepPct: [0, 1],
        controlMode: null,
        id: 'depth',
        instantxControlMode: null,
        kind: 'controlnet',
        model: { base: 'sdxl', hash: 'blake3:depth', key: 'sdxl-depth', name: 'sdxl-depth', type: 'controlnet' },
        modelCondInChannels: undefined,
        modelControlArchitecture: undefined,
        weight: 0.6,
      },
      status: 'contributes',
    });
  });

  describe('regional reference images', () => {
    const asset = { imageName: 'ref.png' } as RegionalGuidanceReferenceImageAsset;
    const ipAdapterModel = { base: 'sd-1', key: 'ipa', name: 'IP Adapter', type: 'ip_adapter' };
    const fluxReduxModel = { base: 'flux', key: 'redux', name: 'FLUX Redux', type: 'flux_redux' };
    const ipAdapterRef = (image: RegionalGuidanceReferenceImageAsset | null): RegionalGuidanceReferenceImage => ({
      config: {
        beginEndStepPct: [0, 1],
        clipVisionModel: 'ViT-H',
        image,
        method: 'full',
        model: ipAdapterModel,
        type: 'ip_adapter',
        weight: 1,
      },
      id: 'ref-ipa',
      isEnabled: true,
    });
    const fluxReduxRef = (image: RegionalGuidanceReferenceImageAsset | null): RegionalGuidanceReferenceImage => ({
      config: { image, imageInfluence: 'highest', model: fluxReduxModel, type: 'flux_redux' },
      id: 'ref-redux',
      isEnabled: true,
    });
    const withModelBase = (base: string): RegionalGuidanceReferenceImage => {
      const ref = ipAdapterRef(asset);
      return { ...ref, config: { ...ref.config, model: { ...ipAdapterModel, base } } };
    };

    it.each<[string, RegionalGuidanceReferenceImage, string, Record<string, unknown>[]]>([
      ['drops an IP-Adapter ref with no image assigned', ipAdapterRef(null), 'sd-1', []],
      [
        'keeps an IP-Adapter ref once an image is assigned for a supported SD base',
        ipAdapterRef(asset),
        'sd-1',
        [{ id: 'ref-ipa', imageName: 'ref.png', type: 'ip_adapter' }],
      ],
      ['drops regional reference images for FLUX.2', ipAdapterRef(asset), 'flux2', []],
      ['drops regional reference images for krea-2', withModelBase('krea-2'), 'krea-2', []],
      ['drops regional reference images for z-image', withModelBase('z-image'), 'z-image', []],
      ['drops regional reference images for anima', withModelBase('anima'), 'anima', []],
      ['keeps an SD2 IP-Adapter ref whose model matches the base', withModelBase('sd-2'), 'sd-2', [{ id: 'ref-ipa' }]],
      ['drops a FLUX Redux ref with no image assigned', fluxReduxRef(null), 'flux', []],
      [
        'sends a FLUX Redux ref its influence as redux settings once an image is assigned',
        fluxReduxRef(asset),
        'flux',
        [{ id: 'ref-redux', settings: { downsampling_factor: 1, weight: 1 }, type: 'flux_redux' }],
      ],
      ['drops a disabled ref even when an image is assigned', { ...ipAdapterRef(asset), isEnabled: false }, 'sd-1', []],
    ])('%s', (_, ref, base, expected) => {
      const verdict = createCanvasInputJudge(judgeParams(main(base), [])).regional(
        region('r', { referenceImages: [ref] })
      );
      expect(verdict.status === 'contributes' ? verdict.input.referenceImages : []).toEqual(
        expected.map((input) => expect.objectContaining(input))
      );
    });
  });
});
