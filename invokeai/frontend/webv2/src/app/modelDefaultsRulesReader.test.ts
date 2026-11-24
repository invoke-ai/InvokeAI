import type { ModelDefaultsRulesState } from '@features/models';

import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { describe, expect, it, vi } from 'vitest';

import { startModelDefaultsRulesReader } from './modelDefaultsRulesReader';

// The served table is seeded directly; the store that fetches it is the one thing stood in for.
vi.mock('@features/models/architecturesStore', () => ({
  ensureArchitectureCapabilitiesLoaded: () => Promise.resolve(),
  getArchitectureCapabilitiesSnapshot: () => ({ revision: 1, status: 'loaded' }),
  subscribeArchitectureCapabilities: () => () => undefined,
}));

seedArchitectureCapabilities();

const vae = (base: string, latent_channels = 32) => ({
  base,
  key: `${base}-vae`,
  latent_channels,
  name: base,
  type: 'vae',
});
const model = (base: string) => ({ base, format: 'checkpoint', key: base, name: base }) as never;

const readReadyState = (): Extract<ModelDefaultsRulesState, { status: 'ready' }> => {
  const published: ModelDefaultsRulesState[] = [];

  startModelDefaultsRulesReader((next) => published.push(next));
  const state = published.at(-1);

  if (state?.status !== 'ready') {
    throw new Error('the rules did not become ready');
  }

  return state;
};

describe('whether Generate loads a default VAE', () => {
  it('is no for a self-contained FLUX.1 SDNQ pipeline, whose own VAE always decodes, and yes for other models', () => {
    const { usesDefaultVae } = readReadyState();
    const pipeline = { text_encoder: {}, text_encoder_2: {}, tokenizer: {}, tokenizer_2: {}, transformer: {}, vae: {} };

    expect(
      usesDefaultVae({
        base: 'flux',
        format: 'sdnq_quantized',
        key: 'flux',
        name: 'flux',
        submodels: pipeline,
      } as never)
    ).toBe(false);
    expect(usesDefaultVae(model('flux'))).toBe(true);
    expect(usesDefaultVae(model('sdxl'))).toBe(true);
  });
});

describe('the default VAE rule the Model Manager reads', () => {
  it("follows the served architecture rows, so a model takes the other base's VAE its row accepts", () => {
    const { acceptsDefaultVae } = readReadyState();

    // MOSS-Image decodes with the FLUX.2 VAE; SDXL's is not among its accepted ones.
    expect(acceptsDefaultVae(model('moss-image'), vae('flux2') as never)).toBe(true);
    expect(acceptsDefaultVae(model('moss-image'), vae('sdxl', 4) as never)).toBe(false);
    // A row that declares no VAE facet accepts its own base only.
    expect(acceptsDefaultVae(model('sdxl'), vae('sdxl', 4) as never)).toBe(true);
    expect(acceptsDefaultVae(model('sdxl'), vae('flux2') as never)).toBe(false);
  });
});
