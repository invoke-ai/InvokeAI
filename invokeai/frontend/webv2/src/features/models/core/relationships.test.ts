import { describe, expect, it } from 'vitest';

import type { RelatableModel } from './relationships';
import type { ModelBase, ModelTaxonomyType } from './types';

import {
  hasLinkableBase,
  isBaseCompatible,
  isLinkableType,
  LINKABLE_TYPES,
  NULL_BASE_ALLOWANCES,
} from './relationships';

const model = (base: ModelBase, type: ModelTaxonomyType = 'lora'): RelatableModel => ({ base, type });

describe('isBaseCompatible', () => {
  it('allows models from the same concrete base', () => {
    expect(isBaseCompatible(model('sdxl', 'main'), model('sdxl'))).toBe(true);
    expect(isBaseCompatible(model('flux', 'main'), model('flux', 'vae'))).toBe(true);
  });

  it('rejects differing concrete bases', () => {
    expect(isBaseCompatible(model('sdxl', 'main'), model('flux'))).toBe(false);
  });

  it('does not treat the null Any base as a universal wildcard', () => {
    expect(isBaseCompatible(model('any'), model('sdxl'))).toBe(false);
    expect(isBaseCompatible(model('sdxl'), model('any'))).toBe(false);
    expect(isBaseCompatible(model('any', 'vae'), model('sdxl', 'main'))).toBe(false);
  });

  it('never links external or unknown bases', () => {
    expect(isBaseCompatible(model('external', 'main'), model('sdxl', 'main'))).toBe(false);
    expect(isBaseCompatible(model('sdxl', 'main'), model('external', 'main'))).toBe(false);
    expect(isBaseCompatible(model('unknown'), model('sdxl'))).toBe(false);
    expect(isBaseCompatible(model('external', 'main'), model('external', 'main'))).toBe(false);
    expect(isBaseCompatible(model('unknown'), model('unknown'))).toBe(false);
  });

  it('allows curated any-based helpers with their consuming bases', () => {
    expect(isBaseCompatible(model('any', 't5_encoder'), model('flux', 'main'))).toBe(true);
    expect(isBaseCompatible(model('any', 't5_encoder'), model('sdxl', 'main'))).toBe(false);
    expect(isBaseCompatible(model('any', 'siglip'), model('flux', 'main'))).toBe(true);
    expect(isBaseCompatible(model('any', 'clip_vision'), model('sdxl', 'main'))).toBe(true);
    expect(isBaseCompatible(model('any', 'mistral_encoder'), model('flux2', 'main'))).toBe(true);
    // Z-Image PiD decode consumes a standalone Gemma2 encoder.
    expect(isBaseCompatible(model('any', 'gemma2_encoder'), model('z-image', 'main'))).toBe(true);
    // No pipeline feeds a CLIP Vision model to an SD2 main.
    expect(isBaseCompatible(model('any', 'clip_vision'), model('sd-2', 'main'))).toBe(false);
  });

  it('is symmetric for every curated entry', () => {
    for (const [type, bases] of Object.entries(NULL_BASE_ALLOWANCES)) {
      if (!bases) {
        continue;
      }

      const helper = model('any', type as ModelTaxonomyType);

      for (const base of ['sd-1', 'sd-2', 'sd-3', 'sdxl', 'flux', 'flux2', 'wan', 'qwen-image', 'z-image'] as const) {
        const host = model(base, 'main');

        expect(isBaseCompatible(helper, host)).toBe(bases.has(base));
        expect(isBaseCompatible(host, helper)).toBe(isBaseCompatible(helper, host));
      }
    }
  });

  it('never links two null-base models', () => {
    expect(isBaseCompatible(model('any', 't5_encoder'), model('any', 'clip_embed'))).toBe(false);
  });
});

describe('hasLinkableBase', () => {
  it('accepts concrete bases and allowanced any-based helpers only', () => {
    expect(hasLinkableBase(model('sdxl', 'main'))).toBe(true);
    expect(hasLinkableBase(model('any', 't5_encoder'))).toBe(true);
    expect(hasLinkableBase(model('any', 'spandrel_image_to_image'))).toBe(false);
    expect(hasLinkableBase(model('external', 'main'))).toBe(false);
    expect(hasLinkableBase(model('unknown', 'main'))).toBe(false);
  });
});

describe('LINKABLE_TYPES', () => {
  it('includes every curated type', () => {
    for (const type of Object.keys(NULL_BASE_ALLOWANCES)) {
      expect(isLinkableType(type as ModelTaxonomyType)).toBe(true);
    }
    expect(LINKABLE_TYPES).toContain('main');
    expect(isLinkableType('onnx')).toBe(false);
  });
});
