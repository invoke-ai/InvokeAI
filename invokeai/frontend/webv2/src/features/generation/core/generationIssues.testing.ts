import type { GenerationIssue } from './generationIssues';
import type { GenerateModelConfig, GenerateReferenceImage, GenerateSettings, MainModelConfig } from './types';

import { resetArchitectureCapabilities, setArchitectureCapabilities } from './architectureCapabilities';
import { architectureCapabilitiesFixture, seedArchitectureCapabilities } from './architectureCapabilities.testing';
import {
  getDefaultGenerateSettings,
  getGenerationModelAvailabilityIssues,
  getGenerationValidationIssues,
  NO_GENERATE_MODEL_ISSUE,
} from './baseGenerationPolicies';

const model = (base: string, overrides: Partial<MainModelConfig> = {}): MainModelConfig => ({
  base,
  key: `${base}-model`,
  name: `${base} model`,
  type: 'main',
  ...overrides,
});
const settingsFor = (main: GenerateModelConfig, overrides: Partial<GenerateSettings> = {}): GenerateSettings => ({
  ...getDefaultGenerateSettings(main),
  seed: 1,
  seedMode: 'fixed',
  ...overrides,
});
const image = { original: { image: { height: 768, image_name: 'reference.png', width: 512 } } };
const reference = (id: string, config: Record<string, unknown>): GenerateReferenceImage => ({
  config: { image, ...config } as GenerateReferenceImage['config'],
  id,
  isEnabled: true,
});
const ipAdapter = (base: string) => ({
  base,
  key: `${base}-ip-adapter`,
  name: `${base} IP Adapter`,
  type: 'ip_adapter',
});
const ipAdapterReference = (id: string, overrides: Record<string, unknown> = {}) =>
  reference(id, {
    beginEndStepPct: [0, 1],
    clipVisionModel: 'ViT-H',
    method: 'full',
    model: ipAdapter('sdxl'),
    type: 'ip_adapter',
    weight: 1,
    ...overrides,
  });

/**
 * Scenarios that together raise every translated Generate issue, for tests that check each against the locale and
 * its renderer. Needs the capability table seeded; leaves it seeded.
 */
export const collectGenerationIssueScenarios = (): GenerationIssue[] => {
  const sdxl = model('sdxl');
  const flux = model('flux');
  const anima = model('anima', { format: 'checkpoint', variant: 'anima_qwen3' });
  const external: GenerateModelConfig = {
    base: 'external',
    capabilities: { max_image_size: { height: 1536, width: 2048 }, modes: ['txt2img'] },
    format: 'external_api',
    key: 'external-model',
    name: 'External Model',
    provider_id: 'openai',
    type: 'external_image_generator',
  };
  const singleReference: GenerateModelConfig = {
    ...external,
    capabilities: { max_reference_images: 1, modes: ['txt2img'], supports_reference_images: true },
  };
  const issues = [
    ...getGenerationValidationIssues(sdxl, settingsFor(sdxl, { height: 1001, width: 40 })),
    ...getGenerationValidationIssues(sdxl, settingsFor(sdxl, { height: 40, width: 1001 })),
    ...getGenerationValidationIssues(external, settingsFor(external, { height: 1536, width: 4096 })),
    ...getGenerationValidationIssues(model('cogview4'), settingsFor(model('cogview4'), { pidMode: 'fit' })),
    ...getGenerationValidationIssues(sdxl, settingsFor(sdxl, { pidMode: 'fit', pidSteps: 0 })),
    ...getGenerationValidationIssues(
      sdxl,
      settingsFor(sdxl, {
        referenceImages: [
          ...Array.from({ length: 5 }, (_, index) => ipAdapterReference(`ref-${index}`)),
          ipAdapterReference('no-image', { image: null }),
          ipAdapterReference('no-adapter', { model: null }),
        ],
      })
    ),
    ...getGenerationValidationIssues(
      singleReference,
      settingsFor(singleReference, {
        referenceImages: [
          reference('a', { type: 'external_reference_image' }),
          reference('b', { type: 'external_reference_image' }),
        ],
      })
    ),
    ...getGenerationValidationIssues(
      model('sd-3'),
      settingsFor(model('sd-3'), { referenceImages: [reference('flux2', { type: 'flux2_reference_image' })] })
    ),
    ...getGenerationValidationIssues(
      flux,
      settingsFor(flux, { referenceImages: [reference('redux', { model: null, type: 'flux_redux' })] })
    ),
    ...getGenerationValidationIssues(
      model('flux', { key: 'kontext', name: 'FLUX Kontext' }),
      settingsFor(model('flux'), {
        referenceImages: [reference('kontext', { model: null, type: 'flux_kontext_reference_image' })],
      })
    ),
    ...getGenerationValidationIssues(
      anima,
      settingsFor(anima, {
        referenceImages: [ipAdapterReference('strength', { model: ipAdapter('anima'), weight: -1 })],
      })
    ),
    ...getGenerationModelAvailabilityIssues(
      flux,
      settingsFor(flux, {
        clipEmbedModel: { base: 'any', key: 'clip', name: 'CLIP Embed', type: 'clip_embed', variant: 'large' },
        loras: [
          { isEnabled: true, model: { base: 'flux', key: 'flux-lora', name: 'FLUX LoRA', type: 'lora' }, weight: 1 },
        ],
        referenceImages: [ipAdapterReference('missing-adapter', { model: ipAdapter('flux') })],
      }),
      []
    ),
    ...getGenerationModelAvailabilityIssues(
      sdxl,
      settingsFor(sdxl, { refinerModel: model('sdxl-refiner', { name: 'SDXL Refiner' }) }),
      [sdxl as never]
    ),
    NO_GENERATE_MODEL_ISSUE,
  ];

  const sdxlSettings = settingsFor(sdxl);

  setArchitectureCapabilities(architectureCapabilitiesFixture.filter((row) => row.base !== 'sdxl'));
  issues.push(...getGenerationValidationIssues(sdxl, sdxlSettings));
  resetArchitectureCapabilities();
  issues.push(...getGenerationValidationIssues(sdxl, sdxlSettings));
  seedArchitectureCapabilities();

  return issues;
};
