import type { BackendGraphContract, BackendInvocationContract } from '@features/generation/contracts';
import type { SupportedGenerateBase } from '@features/generation/core/baseGenerationPolicies';

import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import {
  generateGraphCases,
  satisfiedSettingsFor,
  type ModelShape,
} from '@features/generation/core/graphCoverage.testing';
import { withArchitectureCapabilities } from '@features/generation/graphExport';
import { describe, expect, it } from 'vitest';

import { exportGraphs, GraphExportRequestError, type GraphExportCaseResult } from './graphExport';

const model = (key: string, base: string, type: string, extra: Record<string, unknown> = {}) => ({
  base,
  hash: `blake3:${key}`,
  key,
  name: `${key} name`,
  type,
  ...extra,
});

const MODELS = [
  model('sdxl', 'sdxl', 'main', { format: 'diffusers', variant: 'normal' }),
  model('sdxl-lora', 'sdxl', 'lora', { format: 'lycoris' }),
  model('sdxl-canny', 'sdxl', 'controlnet', { format: 'diffusers' }),
  model('sdxl-ip-adapter', 'sdxl', 'ip_adapter', { format: 'invokeai' }),
  model('z-image-gguf', 'z-image', 'main', { format: 'gguf_quantized' }),
  model('qwen3-06b', 'any', 'qwen3_encoder', { format: 'checkpoint', variant: 'qwen3_06b' }),
  model('qwen3-4b', 'any', 'qwen3_encoder', { format: 'checkpoint', variant: 'qwen3_4b' }),
  model('qwen3-8b', 'any', 'qwen3_encoder', { format: 'checkpoint', variant: 'qwen3_8b' }),
  model('qwen3-4b-gguf', 'any', 'qwen3_encoder', { format: 'gguf_quantized', variant: 'qwen3_4b' }),
  model('sdxl-vae', 'sdxl', 'vae', { format: 'diffusers' }),
  model('flux-vae', 'flux', 'vae', { format: 'checkpoint' }),
];

const sdxlCase = (id: string, overrides: Record<string, unknown> = {}) => ({
  height: 832,
  id,
  mode: 'txt2img',
  model_key: 'sdxl',
  negative_prompt: 'blurry',
  positive_prompt: 'a lighthouse at dusk',
  seed: 4242,
  steps: 12,
  width: 1216,
  ...overrides,
});

const request = (cases: unknown[], models: unknown[] = MODELS) => ({
  capabilities: architectureCapabilitiesFixture,
  cases,
  models,
  version: 1,
});

const exportOne = (testCase: Record<string, unknown>, models?: unknown[]) => {
  const [result] = exportGraphs(request([testCase], models)).cases;

  if (!result || !('batch' in result)) {
    throw new Error(`expected a batch, got ${JSON.stringify(result)}`);
  }

  return result;
};

const nodesOfType = (graph: BackendGraphContract, type: string): BackendInvocationContract[] =>
  Object.values(graph.nodes).filter((node) => node.type === type);

/** A main model of `base` and the components it needs, as graph coverage builds them. */
const satisfiedFor = (base: SupportedGenerateBase, overrides: ModelShape['overrides']) =>
  withArchitectureCapabilities(architectureCapabilitiesFixture, () =>
    satisfiedSettingsFor(base, { label: base, overrides })
  );

const errorOf = (result: GraphExportCaseResult | undefined): string =>
  result && 'error' in result ? result.error : '';

describe('graph export', () => {
  it('sends the case through Generate: seed and prompts as batch data, size and steps in the graph', () => {
    const { batch, id } = exportOne(sdxlCase('sdxl-t2i', { loras: [{ key: 'sdxl-lora', weight: 0.65 }] }));
    const { graph } = batch;
    const noise = nodesOfType(graph, 'noise')[0];
    const denoise = graph.nodes.denoise_latents;

    expect(id).toBe('sdxl-t2i');
    expect(batch.data).toEqual([
      [
        { field_name: 'value', items: [4242], node_path: 'seed' },
        { field_name: 'value', items: ['a lighthouse at dusk'], node_path: 'positive_prompt' },
        { field_name: 'value', items: ['blurry'], node_path: 'negative_prompt' },
      ],
    ]);
    expect(batch.data[0]?.map((datum) => graph.nodes[datum.node_path]?.type)).toEqual(['integer', 'string', 'string']);
    expect(batch).toMatchObject({ destination: 'gallery', origin: 'graphs-export:sdxl-t2i' });
    expect(graph.id).toBe('sdxl-t2i');
    expect([noise?.width, noise?.height, denoise?.steps]).toEqual([1216, 832, 12]);
    expect(
      nodesOfType(graph, 'lora_selector').map((node) => [(node.lora as { key: string }).key, node.weight])
    ).toEqual([['sdxl-lora', 0.65]]);
  });

  it('adds SDXL reference images as full-strength IP-Adapter references with the installed adapter', () => {
    const { graph } = exportOne(sdxlCase('sdxl-refs', { reference_images: ['first.png', 'second.png'] })).batch;

    expect(nodesOfType(graph, 'ip_adapter')).toMatchObject(
      ['first.png', 'second.png'].map((imageName) => ({
        begin_step_percent: 0,
        clip_vision_model: 'ViT-H',
        end_step_percent: 1,
        image: { image_name: imageName },
        ip_adapter_model: { base: 'sdxl', key: 'sdxl-ip-adapter' },
        method: 'full',
        weight: 1,
      }))
    );
  });

  it('adds FLUX.2 reference images as Kontext conditioning, in order', () => {
    const { filled, model: main, settings } = satisfiedFor('flux2', { format: 'diffusers', variant: 'dev' });
    const { graph } = exportOne(
      sdxlCase('flux2-refs', {
        height: settings.height,
        model_key: main.key,
        negative_prompt: null,
        reference_images: ['first.png', 'second.png'],
        width: settings.width,
      }),
      [main, ...filled.map((key) => settings[key])]
    ).batch;
    const kontext = nodesOfType(graph, 'flux_kontext');
    const [firstCollect, lastCollect] = kontext.map(
      (node) => graph.edges.find((edge) => edge.source.node_id === node.id)?.destination.node_id
    );

    expect(kontext.map((node) => node.image)).toEqual([{ image_name: 'first.png' }, { image_name: 'second.png' }]);
    expect(graph.edges).toContainEqual({
      destination: { field: 'collection', node_id: lastCollect },
      source: { field: 'collection', node_id: firstCollect },
    });
    expect(graph.edges).toContainEqual(
      expect.objectContaining({
        destination: expect.objectContaining({ field: 'kontext_conditioning' }),
        source: { field: 'collection', node_id: lastCollect },
      })
    );
  });

  it('makes a lone Krea-2 reference the style reference and several references edit references', () => {
    const { filled, model: main, settings } = satisfiedFor('krea-2', { format: 'diffusers' });
    const krea2Case = (id: string, referenceImages: string[]) =>
      sdxlCase(id, {
        height: settings.height,
        model_key: main.key,
        reference_images: referenceImages,
        width: settings.width,
      });
    const models = [main, ...filled.map((key) => settings[key])];
    const style = exportOne(krea2Case('krea2-style', ['style.png']), models).batch.graph;
    const edit = exportOne(krea2Case('krea2-edit', ['first.png', 'second.png']), models).batch.graph;

    expect(nodesOfType(style, 'krea2_style_reference')).toMatchObject([
      { height: settings.height, image: { image_name: 'style.png' }, style_strength: 1, width: settings.width },
    ]);
    expect(nodesOfType(style, 'qwen_image_i2l')).toEqual([]);
    expect(nodesOfType(edit, 'krea2_style_reference')).toEqual([]);
    expect(nodesOfType(edit, 'qwen_image_i2l').map((node) => [node.image, node.reference])).toEqual([
      [{ image_name: 'first.png' }, true],
      [{ image_name: 'second.png' }, true],
    ]);
  });

  it('rejects reference images for a model without them and more than the model takes', () => {
    const [unsupported, tooMany] = exportGraphs(
      request([
        sdxlCase('z-image-refs', { model_key: 'z-image-gguf', reference_images: ['reference.png'] }),
        sdxlCase('sdxl-six-refs', { reference_images: ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png'] }),
      ])
    ).cases;

    expect(errorOf(unsupported)).toBe('reference_images: z-image-gguf name takes no reference images.');
    expect(errorOf(tooMany)).toBe('sdxl name supports at most 5 reference images. Turn off or remove the extra ones.');
  });

  it('sends img2img through Canvas with the uploaded image and strength', () => {
    const { graph } = exportOne(
      sdxlCase('sdxl-i2i', { init_image: { image_name: 'init.png', strength: 0.3 }, mode: 'img2img' })
    ).batch;

    expect(nodesOfType(graph, 'i2l').map((node) => node.image)).toEqual([{ image_name: 'init.png' }]);
    expect(graph.nodes.denoise_latents?.denoising_start).toBeCloseTo(0.7);
    expect(nodesOfType(graph, 'core_metadata')[0]).toMatchObject({ generation_mode: 'sdxl_img2img', strength: 0.3 });
  });

  it('adds each control entry as a Canvas control layer feeding the denoiser', () => {
    const { graph } = exportOne(
      sdxlCase('sdxl-control', {
        control: [{ begin: 0.1, end: 0.6, image_name: 'canny.png', model_key: 'sdxl-canny', weight: 0.9 }],
        mode: 'controlnet',
      })
    ).batch;
    const [controlNet] = nodesOfType(graph, 'controlnet');

    expect(controlNet).toMatchObject({
      begin_step_percent: 0.1,
      control_model: { key: 'sdxl-canny' },
      control_weight: 0.9,
      end_step_percent: 0.6,
      image: { image_name: 'canny.png' },
    });
    expect(graph.edges).toContainEqual({
      destination: { field: 'control', node_id: 'denoise_latents' },
      source: { field: 'collection', node_id: 'control_net_collector' },
    });
  });

  it('fills required components from the installed models and keeps explicit picks', () => {
    const zImage = { ...sdxlCase('z-image'), model_key: 'z-image-gguf' };

    // The 0.6B encoder is listed first but Z-Image cannot use it.
    expect(exportOne(zImage).components).toEqual({ qwen3EncoderModel: 'qwen3-4b', vae: 'flux-vae' });
    expect(exportOne({ ...zImage, components: { qwen3EncoderModel: 'qwen3-4b-gguf' } }).components).toEqual({
      qwen3EncoderModel: 'qwen3-4b-gguf',
      vae: 'flux-vae',
    });
  });

  // The working-memory calibration planner (`default_vae`) restates this: it sizes each case for the VAE that decodes.
  it("decodes with the main model's default VAE, which a pick replaces where the model has a VAE slot", () => {
    const withDefaultVae = (key: string, vaeKey: string) => [
      ...MODELS.map((entry) => (entry.key === key ? { ...entry, default_settings: { vae: vaeKey } } : entry)),
      model('flux-vae-b', 'flux', 'vae', { format: 'checkpoint' }),
    ];
    const { batch } = exportOne(sdxlCase('sdxl-default-vae'), withDefaultVae('sdxl', 'sdxl-vae'));
    const zImage = { ...sdxlCase('z-image'), model_key: 'z-image-gguf' };
    const zImageModels = withDefaultVae('z-image-gguf', 'flux-vae-b');

    expect(nodesOfType(batch.graph, 'vae_loader').map((node) => (node.vae_model as { key: string }).key)).toEqual([
      'sdxl-vae',
    ]);
    expect(exportOne(zImage, zImageModels).components.vae).toBe('flux-vae-b');
    expect(exportOne({ ...zImage, components: { vae: 'flux-vae' } }, zImageModels).components.vae).toBe('flux-vae');
  });

  it('rejects an installed component the slot does not offer', () => {
    const [result] = exportGraphs(
      request([sdxlCase('z-image', { components: { qwen3EncoderModel: 'qwen3-8b' }, model_key: 'z-image-gguf' })])
    ).cases;

    expect(errorOf(result)).toBe(
      'components.qwen3EncoderModel: "qwen3-8b" cannot be the Qwen3 Encoder of z-image-gguf name. ' +
        'Choices: qwen3-4b, qwen3-4b-gguf.'
    );
  });

  it.each(generateGraphCases)('compiles $label from exactly the components it needs', ({ base, shape }) => {
    const {
      filled,
      model: main,
      settings,
    } = withArchitectureCapabilities(architectureCapabilitiesFixture, () => satisfiedSettingsFor(base, shape));
    const components = Object.fromEntries(filled.map((key) => [key, (settings[key] as { key: string }).key]));
    const testCase = sdxlCase(base, { height: settings.height, model_key: main.key, width: settings.width });

    expect(exportOne(testCase, [main, ...filled.map((key) => settings[key])]).components).toEqual(components);
  });

  it('applies settings overrides and rejects values Generate would not keep', () => {
    const { graph } = exportOne(sdxlCase('cfg', { settings: { cfgScale: 3.5, scheduler: 'dpmpp_2m' } })).batch;
    const [badPid, badScheduler] = exportGraphs(
      request([
        sdxlCase('pid', { settings: { pidMode: 'sideways' } }),
        sdxlCase('scheduler', { settings: { scheduler: 'not-a-scheduler' } }),
      ])
    ).cases;

    expect(graph.nodes.denoise_latents).toMatchObject({ cfg_scale: 3.5, scheduler: 'dpmpp_2m' });
    expect(errorOf(badPid)).toMatch(/^settings\.pidMode: Generate does not keep "sideways"/);
    expect(errorOf(badScheduler)).toMatch(/^settings\.scheduler: .* Options: .*euler_a/);
  });

  it('rejects a prompt with dynamic syntax while dynamic prompts are on, and sends it as typed when they are off', () => {
    const prompt = 'a {red|green} lighthouse';
    const [expanding] = exportGraphs(request([sdxlCase('on', { positive_prompt: prompt })])).cases;
    const { batch } = exportOne(
      sdxlCase('off', { positive_prompt: prompt, settings: { dynamicPromptsEnabled: false } })
    );

    expect(errorOf(expanding)).toMatch(/^positive_prompt has dynamic prompt syntax/);
    expect(batch.data[0]).toContainEqual({ field_name: 'value', items: [prompt], node_path: 'positive_prompt' });
  });

  it('reports unknown names per case, with the valid choices, and still exports the other cases', () => {
    const response = exportGraphs(
      request([
        sdxlCase('unknown-setting', { settings: { cfgscale: 3 } }),
        sdxlCase('unknown-slot', { components: { t5EncoderModel: 'qwen3-4b' }, model_key: 'z-image-gguf' }),
        sdxlCase('unknown-component', { components: { vae: 'missing-vae' }, model_key: 'z-image-gguf' }),
        sdxlCase('unknown-model', { model_key: 'missing' }),
        sdxlCase('unknown-lora', { loras: [{ key: 'missing-lora', weight: 1 }] }),
        sdxlCase('unknown-control', {
          control: [{ begin: 0, end: 1, image_name: 'canny.png', model_key: 'sdxl-lora', weight: 1 }],
          mode: 'controlnet',
        }),
        sdxlCase('ok'),
      ])
    );
    const errors = Object.fromEntries(response.cases.map((result) => [result.id, errorOf(result)]));

    expect(errors['unknown-setting']).toMatch(/^Unknown settings key "cfgscale"\. Valid keys: .*\bcfgScale\b/);
    expect(errors['unknown-slot']).toMatch(
      /^Unknown component slot "t5EncoderModel" .* Valid slots: .*qwen3EncoderModel, vae/
    );
    expect(errors['unknown-component']).toBe(
      'components.vae: model key "missing-vae" is not installed. Choices: flux-vae.'
    );
    expect(errors['unknown-model']).toBe(
      'model_key "missing" is not a selectable main model. Choices: sdxl, z-image-gguf.'
    );
    expect(errors['unknown-lora']).toMatch(/^loras\[0\]: "missing-lora" .* Choices: sdxl-lora\.$/);
    expect(errors['unknown-control']).toMatch(/^control\[0\]: "sdxl-lora" cannot drive .* Choices: sdxl-canny\.$/);
    expect(response.cases.find((result) => result.id === 'ok')).toHaveProperty('batch');
  });

  it('exports identical requests identically, with stable node ids', () => {
    const cases = [
      sdxlCase('t2i', { loras: [{ key: 'sdxl-lora', weight: 1 }], reference_images: ['reference.png'] }),
      sdxlCase('i2i', { init_image: { image_name: 'init.png', strength: 0.5 }, mode: 'img2img' }),
      { ...sdxlCase('z-image'), model_key: 'z-image-gguf' },
    ];
    const first = exportGraphs(request(cases));
    const nodeIds = first.cases.flatMap((result) => ('batch' in result ? Object.keys(result.batch.graph.nodes) : []));

    expect(exportGraphs(request(cases))).toEqual(first);
    expect(nodeIds.filter((id) => /_[a-z0-9]{7,9}_[a-z0-9]{4,6}$/.test(id))).toEqual([]);
  });

  it.each([
    ['a non-object request', []],
    ['another version', { ...request([]), version: 2 }],
    ['models that are not model configs', request([], [{ key: 'only-a-key' }])],
    ['cases without ids', request([{ mode: 'txt2img' }])],
    ['duplicate case ids', request([sdxlCase('same'), sdxlCase('same')])],
  ])('rejects %s as a malformed request', (_label, malformed) => {
    expect(() => exportGraphs(malformed)).toThrow(GraphExportRequestError);
  });
});
