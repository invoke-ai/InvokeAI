import type { GenerateModelConfig, MainModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type {
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasRegionalGuidanceLayerContract,
  RegionalGuidanceReferenceImage,
  RegionalGuidanceReferenceImageAsset,
} from '@workbench/canvas-engine/contracts';

import { resetArchitectureCapabilities } from '@features/generation/core/architectureCapabilities';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
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

/** What Invoke readiness reports for these layers, every region inside the frame. */
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

/** What the invocation decides for each layer composition asks it about, in generation order. */
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

    // Before the model list loads too: a hosted model takes no layer whatever is installed.
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
    // The region it can place still needs a global prompt to lay out around.
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
  const models = [model('sdxl-control', 'sdxl', 'controlnet'), model('sd1-control', 'sd-1', 'controlnet')];
  const controlIssues = (
    layers: CanvasLayerContract[],
    mainModel: GenerateModelConfig,
    installed: readonly ModelConfig[],
    controlModelMemory?: Record<string, string>
  ) => readinessIssues(layers, judgeParams(mainModel, installed, controlModelMemory));

  it('names raster content in the frame first for a text-only hosted model', () => {
    const editing: GenerateModelConfig = { ...hostedModel, capabilities: { modes: ['txt2img', 'img2img'] } };

    expect(readinessIssues([controlLayer('guide')], judgeParams(hostedModel, []), { rasterInFrame: true })).toEqual([
      { kind: 'hosted_refusal', modelName: 'OpenAI Image', refusal: { kind: 'raster_content' } },
      { kind: 'hosted_refusal', modelName: 'OpenAI Image', refusal: { kind: 'control_layer', layerName: 'guide' } },
    ]);
    expect(readinessIssues([], judgeParams(editing, []), { rasterInFrame: true })).toEqual([]);
  });

  it('leaves out a content-bearing layer with neither a type nor a model: it warns but contributes nothing', () => {
    expect(controlIssues([controlLayer('Control Layer 1')], sdxl, models)).toEqual([]);
  });

  it('blocks an untyped layer whose model is no longer installed', () => {
    expect(controlIssues([controlLayer('legacy', { model: 'uninstalled' })], sdxl, models)).toMatchObject([
      { code: 'missing_model', layerId: 'legacy' },
    ]);
  });

  it('reports a typed content-bearing layer no model resolves', () => {
    expect(
      controlIssues(
        [controlLayer('Control Layer 1', { derivation: { controlType: 'pose', detection: null } })],
        sdxl,
        models
      )
    ).toEqual([
      { code: 'no_model_for_type', kind: 'control_layer', layerId: 'Control Layer 1', layerName: 'Control Layer 1' },
    ]);
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

  it('reports an incompatible base model', () => {
    expect(controlIssues([controlLayer('mismatch', { model: 'sd1-control' })], sdxl, models)).toMatchObject([
      { code: 'incompatible_base' },
    ]);
  });

  it('reports Anima LLLite layers the denoiser would refuse, and only those', () => {
    const animaModels = [
      { ...model('sketch', 'anima', 'controlnet'), cond_in_channels: 3 },
      { ...model('depth', 'anima', 'controlnet'), cond_in_channels: 3 },
      { ...model('inpaint', 'anima', 'controlnet'), cond_in_channels: 4 },
    ];
    const lllite = (id: string, key: string) => controlLayer(id, { kind: 'anima_lllite', model: key });

    expect(
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
      ).map((issue) => (issue.kind === 'control_layer' ? [issue.layerId, issue.code] : issue))
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
    const codes = (mainModel: GenerateModelConfig) =>
      controlIssues(layers, mainModel, union).map((issue) =>
        issue.kind === 'control_layer' ? [issue.layerId, issue.code] : issue
      );

    expect(codes(main(base, base === 'flux2' ? 'dev' : undefined))).toEqual([['second', limitCode]]);
    expect(codes(main(base, refusingVariant))).toEqual([
      ['first', refusalCode],
      ['second', refusalCode],
    ]);
  });

  it('lets one Z-Image control through', () => {
    const layers = ['first', 'second'].map((id) => controlLayer(id, { kind: 'z_image_control', model: 'z-control' }));

    expect(
      controlIssues(layers, main('z-image'), [model('z-control', 'z-image', 'controlnet')]).map((issue) =>
        issue.kind === 'control_layer' ? [issue.layerId, issue.code] : issue
      )
    ).toEqual([['second', 'z_image_control_limit']]);
  });

  it('reports issues in document order', () => {
    const pose = { derivation: { controlType: 'pose', detection: null } } as const;

    expect(
      controlIssues([controlLayer('top', pose), controlLayer('bottom', pose)], sdxl, models).map((issue) =>
        issue.kind === 'control_layer' ? issue.layerName : issue
      )
    ).toEqual(['top', 'bottom']);
  });

  describe('typed control layers', () => {
    const depthModels = [model('sdxl-depth-a', 'sdxl', 'controlnet'), model('sdxl-depth-b', 'sdxl', 'controlnet')];
    const typed = (
      id: string,
      derivation: Partial<Pick<CanvasControlLayerContract, 'detection' | 'filter' | 'original'>> = {},
      overrides: Parameters<typeof controlLayer>[1] = {}
    ) => controlLayer(id, { ...overrides, derivation: { controlType: 'depth', detection: null, ...derivation } });

    it('asks for a model when several fit, and resolves with the remembered pick', () => {
      expect(controlIssues([typed('depth')], sdxl, depthModels)).toMatchObject([{ code: 'choose_model' }]);
      expect(controlIssues([typed('depth')], sdxl, depthModels, { 'sdxl:depth': 'sdxl-depth-b' })).toEqual([]);
    });

    it('tells no control models apart from none for the type', () => {
      expect(controlIssues([typed('depth')], sdxl, [])).toMatchObject([{ code: 'no_control_models' }]);
    });

    it('never blocks on an empty layer', () => {
      expect(
        controlIssues([typed('empty', {}, { source: { bitmap: null, type: 'paint' } })], sdxl, depthModels)
      ).toEqual([]);
    });

    it('blocks on a map detection has not caught up with, after the model reasons', () => {
      const pending = typed('pending', { detection: { settings: {}, type: 'depth_anything_depth_estimation' } });

      expect(controlIssues([pending], sdxl, [depthModels[0]!])).toMatchObject([{ code: 'detection_pending' }]);
      expect(controlIssues([pending], sdxl, depthModels)).toMatchObject([{ code: 'choose_model' }]);
    });

    it('counts per-generation limits by the kind the resolved model runs, not the stored kind', () => {
      expect(
        controlIssues([typed('first'), typed('second')], main('flux'), [
          model('flux-depth-lora', 'flux', 'control_lora'),
        ]).map((issue) => (issue.kind === 'control_layer' ? [issue.layerId, issue.code] : issue))
      ).toEqual([['second', 'control_lora_limit']]);
    });

    it('still resolves models before capabilities load, and reports that they have not', () => {
      const judge = () =>
        controlIssues([typed('depth')], main('flux'), [model('flux-depth-t2i', 'flux', 't2i_adapter')]);

      // Loaded, FLUX runs no T2I-Adapter, so no model fits.
      expect(judge()).toMatchObject([{ code: 'no_control_models' }]);
      resetArchitectureCapabilities();
      expect(judge()).toMatchObject([{ code: 'capabilities_unavailable' }]);
    });
  });
});

describe('createCanvasInputJudge', () => {
  it('hands a contributing control layer its adapter values and the model it resolves to', () => {
    const depth = { ...model('sdxl-depth', 'sdxl', 'controlnet'), hash: 'blake3:depth' };
    const layer = controlLayer('depth', {
      derivation: { controlType: 'depth', detection: null },
      weight: 0.6,
    });

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

  it('gives a layer a FLUX union model resolves the InstantX mode of its type', () => {
    const union = model('flux-union', 'flux', 'controlnet');
    const layer = controlLayer('pose', { derivation: { controlType: 'pose', detection: null } });

    expect(
      createCanvasInputJudge(
        judgeParams(main('flux'), [{ ...union, name: 'FLUX.1-dev-Controlnet-Union-Pro' }])
      ).control(layer)
    ).toMatchObject({ input: { instantxControlMode: 4, model: { key: 'flux-union' } }, status: 'contributes' });
  });

  // A region's reference images reach the graph only with an assigned image and a model of the main base.
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
      config: { image, imageInfluence: 'high', model: fluxReduxModel, type: 'flux_redux' },
      id: 'ref-redux',
      isEnabled: true,
    });
    /** The reference images a captioned region on this base contributes. */
    const referencesFor = (base: string, referenceImages: RegionalGuidanceReferenceImage[]) => {
      const verdict = createCanvasInputJudge(judgeParams(main(base), [])).regional(region('r', { referenceImages }));
      return verdict.status === 'contributes' ? verdict.input.referenceImages : verdict;
    };

    it('drops an IP-Adapter ref with no image assigned, and keeps it once one is', () => {
      expect(referencesFor('sd-1', [ipAdapterRef(null)])).toEqual([]);
      expect(referencesFor('sd-1', [ipAdapterRef(asset)])).toEqual([
        {
          beginEndStepPct: [0, 1],
          clipVisionModel: 'ViT-H',
          id: 'ref-ipa',
          imageName: 'ref.png',
          method: 'full',
          model: ipAdapterModel,
          type: 'ip_adapter',
          weight: 1,
        },
      ]);
    });

    it.each(['flux2', 'krea-2', 'z-image', 'anima'])('drops regional reference images for %s', (base) => {
      const ref = ipAdapterRef(asset);
      expect(referencesFor(base, [{ ...ref, config: { ...ref.config, model: { ...ipAdapterModel, base } } }])).toEqual(
        []
      );
    });

    it('keeps an SD2 IP-Adapter ref whose model matches the base', () => {
      const ref = ipAdapterRef(asset);
      expect(
        referencesFor('sd-2', [{ ...ref, config: { ...ref.config, model: { ...ipAdapterModel, base: 'sd-2' } } }])
      ).toHaveLength(1);
    });

    it('sends a FLUX Redux ref its influence as redux settings once an image is assigned', () => {
      expect(referencesFor('flux', [fluxReduxRef(null)])).toEqual([]);
      expect(referencesFor('flux', [fluxReduxRef(asset)])).toEqual([
        {
          id: 'ref-redux',
          imageName: 'ref.png',
          model: fluxReduxModel,
          settings: { downsampling_factor: 2, weight: 1 },
          type: 'flux_redux',
        },
      ]);
    });

    it('drops a disabled ref even when an image is assigned', () => {
      expect(referencesFor('sd-1', [{ ...ipAdapterRef(asset), isEnabled: false }])).toEqual([]);
    });
  });
});
