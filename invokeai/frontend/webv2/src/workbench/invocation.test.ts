import { describe, expect, it } from 'vitest';

import type {
  ComponentModelConfig,
  GenerateModelConfig,
  GenerateWidgetValues,
  MainModelConfig,
  VaeModelConfig,
} from './generation/types';

import { getDefaultGenerateSettings } from './generation/baseGenerationPolicies';
import { resolveInvocationRoute } from './invocation';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState';

const animaModel: MainModelConfig = { base: 'anima', key: 'anima-model', name: 'Anima', type: 'main' };
const animaVae: VaeModelConfig = { base: 'qwen-image', key: 'anima-vae', name: 'Anima VAE', type: 'vae' };
const flux2Model: MainModelConfig = {
  base: 'flux2',
  format: 'gguf_quantized',
  key: 'flux2-klein-9b',
  name: 'FLUX.2 Klein 9B',
  type: 'main',
  variant: 'klein_9b',
};
const incompatibleFlux2Source: MainModelConfig = {
  base: 'flux2',
  format: 'diffusers',
  key: 'flux2-klein-4b-source',
  name: 'FLUX.2 Klein 4B Source',
  type: 'main',
  variant: 'klein_4b',
};
const flux2Vae: VaeModelConfig = { base: 'flux2', key: 'flux2-vae', name: 'FLUX.2 VAE', type: 'vae' };
const flux2Qwen3Encoder: ComponentModelConfig = {
  base: 'any',
  key: 'flux2-qwen3-encoder',
  name: 'Qwen3 8B Encoder',
  type: 'qwen3_encoder',
  variant: 'qwen3_8b',
};
const qwen3Encoder: ComponentModelConfig = {
  base: 'any',
  key: 'qwen3-encoder',
  name: 'Qwen3 Encoder',
  type: 'qwen3_encoder',
  variant: 'qwen3_06b',
};
const unknownExternalModel: GenerateModelConfig = {
  base: 'external',
  capabilities: { modes: ['txt2img'] },
  format: 'external_api',
  key: 'external-unknown',
  name: 'Unknown External Provider',
  provider_id: 'future-provider',
  type: 'external_image_generator',
};

const createGenerateValues = (
  model: GenerateModelConfig = animaModel,
  overrides: Partial<GenerateWidgetValues> = {}
): GenerateWidgetValues => ({
  ...getDefaultGenerateSettings(model),
  model,
  modelKey: model.key,
  positivePrompt: 'an anima prompt',
  ...overrides,
});

const getActiveProject = (values: GenerateWidgetValues) => {
  const state = workbenchReducer(createInitialWorkbenchState(), { type: 'setGenerateSettings', values });
  const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

  expect(project).toBeDefined();

  return project!;
};

describe('resolveInvocationRoute', () => {
  it('invalidates Anima generation until required components are selected', () => {
    const missingComponentsRoute = resolveInvocationRoute(
      getActiveProject(createGenerateValues(animaModel, { vae: animaVae }))
    );

    expect(missingComponentsRoute.sourceValid).toBe(false);
    expect(missingComponentsRoute.validationReasons).toContain('Generate needs a Qwen3 Encoder for Anima models.');

    const validRoute = resolveInvocationRoute(
      getActiveProject(createGenerateValues(animaModel, { qwen3EncoderModel: qwen3Encoder, vae: animaVae }))
    );

    expect(validRoute.sourceValid).toBe(true);
    expect(validRoute.validationReasons).toEqual([]);
  });

  it('invalidates non-Diffusers FLUX.2 generation when a stale component source cannot provide the required Qwen3 encoder', () => {
    const route = resolveInvocationRoute(
      getActiveProject(
        createGenerateValues(flux2Model, { componentSourceModel: incompatibleFlux2Source, vae: flux2Vae })
      )
    );

    expect(route.sourceValid).toBe(false);
    expect(route.validationReasons).toContain('Generate needs a Qwen3 Encoder for non-Diffusers FLUX.2 models.');
  });

  it('invalidates FLUX.2 generation when dimensions are not on the required 16px grid', () => {
    const route = resolveInvocationRoute(
      getActiveProject(
        createGenerateValues(flux2Model, { height: 888, qwen3EncoderModel: flux2Qwen3Encoder, vae: flux2Vae })
      )
    );

    expect(route.sourceValid).toBe(false);
    expect(route.validationReasons).toContain('Generate height must be a multiple of 16.');
  });

  it('invalidates external providers without a registered invocation node', () => {
    const route = resolveInvocationRoute(getActiveProject(createGenerateValues(unknownExternalModel)));

    expect(route.sourceValid).toBe(false);
    expect(route.validationReasons).toContain("No invocation node registered for external provider 'future-provider'.");
  });
});
