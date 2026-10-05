import { describe, expect, it } from 'vitest';

import type { RelatableModel } from './relationships';
import type { ModelBase, ModelTaxonomyType } from './types';

import {
  CROSS_BASE_ALLOWANCES,
  hasLinkableBase,
  isBaseCompatible,
  isLinkablePair,
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

  it('allows curated cross-base helpers with their consuming bases', () => {
    // z_image_model_loader takes a FLUX VAE; krea2_model_loader takes Qwen-Image or Anima VAEs.
    expect(isBaseCompatible(model('z-image', 'main'), model('flux', 'vae'))).toBe(true);
    expect(isBaseCompatible(model('krea-2', 'main'), model('qwen-image', 'vae'))).toBe(true);
    expect(isBaseCompatible(model('krea-2', 'main'), model('anima', 'vae'))).toBe(true);
    // flux2_klein_model_loader takes a FLUX VAE; anima_model_loader takes Wan/Qwen-Image/FLUX VAEs.
    expect(isBaseCompatible(model('flux2', 'main'), model('flux', 'vae'))).toBe(true);
    expect(isBaseCompatible(model('anima', 'main'), model('wan', 'vae'))).toBe(true);
    expect(isBaseCompatible(model('anima', 'main'), model('qwen-image', 'vae'))).toBe(true);
    expect(isBaseCompatible(model('anima', 'main'), model('flux', 'vae'))).toBe(true);
    // z_image_pid_decode reuses the FLUX PiD decoder.
    expect(isBaseCompatible(model('z-image', 'main'), model('flux', 'pid_decoder'))).toBe(true);
  });

  it('limits cross-base allowances to the helper type and cited pairs', () => {
    // The allowance belongs to the VAE type, not to the base pair.
    expect(isBaseCompatible(model('z-image', 'main'), model('flux', 'lora'))).toBe(false);
    expect(isBaseCompatible(model('z-image', 'main'), model('flux', 'main'))).toBe(false);
    // No loader feeds an SD-3 VAE to a Z-Image pipeline.
    expect(isBaseCompatible(model('z-image', 'main'), model('sd-3', 'vae'))).toBe(false);
    // Only the FLUX decoder is shared; other PiD pairings must match exactly.
    expect(isBaseCompatible(model('qwen-image', 'main'), model('flux', 'pid_decoder'))).toBe(false);
    expect(isBaseCompatible(model('sdxl', 'main'), model('sdxl', 'pid_decoder'))).toBe(true);
  });

  it('is symmetric for every cross-base entry', () => {
    for (const [type, byBase] of Object.entries(CROSS_BASE_ALLOWANCES)) {
      for (const [helperBase, hosts] of Object.entries(byBase ?? {})) {
        for (const hostBase of hosts ?? []) {
          const helper = model(helperBase as ModelBase, type as ModelTaxonomyType);
          const host = model(hostBase, 'main');

          expect(isBaseCompatible(helper, host)).toBe(true);
          expect(isBaseCompatible(host, helper)).toBe(true);
        }
      }
    }
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

describe('isLinkablePair', () => {
  it('blocks same-type pairs of single-slot types', () => {
    // A pipeline runs exactly one of each; two are never "used together".
    expect(isLinkablePair(model('sdxl', 'main'), model('sdxl', 'main'))).toBe(false);
    expect(isLinkablePair(model('sdxl', 'vae'), model('sdxl', 'vae'))).toBe(false);
    expect(isLinkablePair(model('any', 't5_encoder'), model('any', 't5_encoder'))).toBe(false);
    expect(isLinkablePair(model('flux', 'pid_decoder'), model('flux', 'pid_decoder'))).toBe(false);
  });

  it('keeps stackable same-type pairs linkable', () => {
    expect(isLinkablePair(model('sdxl', 'lora'), model('sdxl', 'lora'))).toBe(true);
    expect(isLinkablePair(model('sdxl', 'embedding'), model('sdxl', 'embedding'))).toBe(true);
    expect(isLinkablePair(model('sdxl', 'ip_adapter'), model('sdxl', 'ip_adapter'))).toBe(true);
    expect(isLinkablePair(model('sdxl', 't2i_adapter'), model('sdxl', 't2i_adapter'))).toBe(true);
  });

  it('defers to base compatibility for cross-type pairs', () => {
    expect(isLinkablePair(model('sdxl', 'main'), model('sdxl', 'vae'))).toBe(true);
    expect(isLinkablePair(model('z-image', 'main'), model('flux', 'vae'))).toBe(true);
    expect(isLinkablePair(model('sdxl', 'main'), model('flux', 'vae'))).toBe(false);
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

  it('includes the concrete-based helper types the backend loaders consume', () => {
    // flux_control_lora_loader, flux_redux, and pid_decoder_loader.
    expect(isLinkableType('control_lora')).toBe(true);
    expect(isLinkableType('flux_redux')).toBe(true);
    expect(isLinkableType('pid_decoder')).toBe(true);
  });
});
