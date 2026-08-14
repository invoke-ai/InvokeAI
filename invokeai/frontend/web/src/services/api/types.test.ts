import { describe, expect, it } from 'vitest';

import { isFlux2DiffusersMainModelConfig, isZImageDiffusersMainModelConfig } from './types';

const partialConfig = (base: 'flux2' | 'z-image', submodels: Record<string, unknown>) => ({
  type: 'main',
  base,
  format: 'sdnq_quantized',
  variant: 'klein_4b',
  name: 'partial-sdnq-pipeline',
  submodels,
});

describe('SDNQ pipeline model predicates', () => {
  it.each([
    ['flux2', isFlux2DiffusersMainModelConfig],
    ['z-image', isZImageDiffusersMainModelConfig],
  ] as const)('rejects a pipeline with only a transformer submodel', (base, predicate) => {
    expect(predicate(partialConfig(base, { transformer: {} }) as never)).toBe(false);
  });

  it.each([
    ['flux2', isFlux2DiffusersMainModelConfig],
    ['z-image', isZImageDiffusersMainModelConfig],
  ] as const)('rejects a pipeline with no transformer submodel', (base, predicate) => {
    expect(predicate(partialConfig(base, { vae: {}, text_encoder: {}, tokenizer: {} }) as never)).toBe(false);
  });
});
