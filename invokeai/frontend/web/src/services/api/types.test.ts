import { describe, expect, it } from 'vitest';

import type { MainModelConfig } from './types';
import {
  isFlux2DiffusersMainModelConfig,
  isWanLowNoisePartnerOption,
  isZImageDiffusersMainModelConfig,
  selectPrimaryMainModelOptions,
} from './types';

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

const wanMain = (over: Record<string, unknown>) =>
  ({
    key: 'k',
    type: 'main',
    base: 'wan',
    format: 'checkpoint',
    variant: 't2v_a14b',
    name: 'wan',
    ...over,
  }) as unknown as MainModelConfig;

describe('Wan low-noise partner picker', () => {
  it('offers an untagged single-file A14B', () => {
    // The case the whole branch exists to support. `expert` comes from a filename
    // heuristic, is absent from `ModelRecordChanges`, and installed records are never
    // re-probed — so requiring `expert === 'low'` here strands untagged pairs
    // permanently, with no correction short of delete-and-reinstall.
    expect(isWanLowNoisePartnerOption(wanMain({ key: 'untagged', expert: 'none' }))).toBe(true);
  });

  it('offers a tagged low expert, and a GGUF one — the pair need not share a format', () => {
    expect(isWanLowNoisePartnerOption(wanMain({ key: 'low', expert: 'low' }))).toBe(true);
    expect(isWanLowNoisePartnerOption(wanMain({ key: 'low-gguf', expert: 'low', format: 'gguf_quantized' }))).toBe(
      true
    );
  });

  it('does not offer a tagged high expert or a Diffusers main', () => {
    expect(isWanLowNoisePartnerOption(wanMain({ key: 'high', expert: 'high' }))).toBe(false);
    expect(isWanLowNoisePartnerOption(wanMain({ key: 'diff', format: 'diffusers' }))).toBe(false);
  });

  it('does not offer a TI2V-5B — it is single-transformer and has no partner', () => {
    expect(isWanLowNoisePartnerOption(wanMain({ key: '5b', variant: 'ti2v_5b', expert: 'none' }))).toBe(false);
  });

  it('lets an untagged pair be assembled: both halves stay in the primary picker too', () => {
    // The two pickers have to agree. Widening this one must not start hiding untagged
    // models from the primary list — `selectPrimaryMainModelOptions` keys on the narrow
    // tag test for exactly that reason.
    const a = wanMain({ key: 'a', expert: 'none', name: 'pair-part-1' });
    const b = wanMain({ key: 'b', expert: 'none', name: 'pair-part-2' });

    expect([a, b].filter(isWanLowNoisePartnerOption)).toHaveLength(2);
    expect(selectPrimaryMainModelOptions([a, b])).toHaveLength(2);
  });

  it('keeps an untagged model in the primary picker even next to a tagged high expert', () => {
    // The case that actually catches `selectPrimaryMainModelOptions` being switched to the
    // wide predicate. With two untagged models the wide test classes both as low experts,
    // so neither has a partner and neither is hidden — the mistake hides behind itself.
    // Add a same-variant `high` and the untagged model suddenly has a partner, so keying
    // the primary filter on the wide test would drop it from the main picker entirely.
    const high = wanMain({ key: 'high', expert: 'high', name: 'high' });
    const untagged = wanMain({ key: 'untagged', expert: 'none', name: 'untagged' });

    expect(selectPrimaryMainModelOptions([high, untagged]).map((c) => c.key)).toEqual(['high', 'untagged']);
  });

  it('still hides a tagged low expert from the primary picker when it has a partner', () => {
    const high = wanMain({ key: 'high', expert: 'high', name: 'high' });
    const low = wanMain({ key: 'low', expert: 'low', name: 'low' });

    expect(selectPrimaryMainModelOptions([high, low]).map((c) => c.key)).toEqual(['high']);
  });
});
