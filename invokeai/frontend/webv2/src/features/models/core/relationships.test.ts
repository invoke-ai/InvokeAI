import { describe, expect, it } from 'vitest';

import type { RelatableModel } from './relationships';
import type { ModelBase, ModelTaxonomyType } from './types';

import { seedArchitectureCapabilities } from './architectureCapabilities.testing';
import {
  CROSS_BASE_ALLOWANCES,
  hasLinkableBase,
  isBaseCompatible,
  isLinkablePair,
  isLinkableModel,
  isLinkableType,
  LINKABLE_TYPES,
  NULL_BASE_ALLOWANCES,
} from './relationships';

const model = (base: ModelBase, type: ModelTaxonomyType = 'lora'): RelatableModel => ({ base, type });
const main = (base: ModelBase, variant?: string): RelatableModel => ({ base, type: 'main', variant });
const vae = (base: ModelBase, latentChannels?: number): RelatableModel => ({
  base,
  latent_channels: latentChannels,
  type: 'vae',
});

describe('isBaseCompatible', () => {
  it('allows models from the same concrete base', () => {
    expect(isBaseCompatible(model('sdxl', 'main'), model('sdxl'))).toBe(true);
  });

  it('rejects differing concrete bases', () => {
    expect(isBaseCompatible(model('sdxl', 'main'), model('flux'))).toBe(false);
  });

  it('allows curated cross-base helpers with their consuming bases', () => {
    expect(isBaseCompatible(model('z-image', 'main'), model('flux', 'pid_decoder'))).toBe(true);
    // MOSS-Image decodes with flux2_pid_decode.
    expect(isBaseCompatible(model('moss-image', 'main'), model('flux2', 'pid_decoder'))).toBe(true);
    expect(isBaseCompatible(model('moss-image', 'main'), model('flux2', 'lora'))).toBe(false);
  });

  it('limits cross-base allowances to the helper type and cited pairs', () => {
    expect(isBaseCompatible(model('z-image', 'main'), model('flux', 'lora'))).toBe(false);
    expect(isBaseCompatible(model('z-image', 'main'), model('flux', 'main'))).toBe(false);
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
    expect(isBaseCompatible(model('any', 'gemma2_encoder'), model('z-image', 'main'))).toBe(true);
    expect(isBaseCompatible(model('any', 'qwen3_5_encoder'), model('moss-image', 'main'))).toBe(true);
    expect(isBaseCompatible(model('any', 'gemma2_encoder'), model('moss-image', 'main'))).toBe(true);
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
    expect(isLinkablePair(model('sdxl', 'main'), model('sdxl', 'lora'))).toBe(true);
    expect(isLinkablePair(model('z-image', 'main'), model('flux', 'pid_decoder'))).toBe(true);
    expect(isLinkablePair(model('sdxl', 'main'), model('flux', 'lora'))).toBe(false);
  });
});

describe('VAE pairs', () => {
  // The served table decides; these expectations follow the backend's VAE facets, not a copy kept here.
  seedArchitectureCapabilities();

  it("follow the host architecture's served VAE facet across bases", () => {
    expect(isLinkablePair(main('sdxl'), vae('sdxl'))).toBe(true);
    expect(isLinkablePair(main('ernie-image'), vae('flux2'))).toBe(true);
    expect(isLinkablePair(main('qwen-image'), vae('anima'))).toBe(true);
    // A FLUX VAE decodes an Anima latent without raising and returns a smear.
    expect(isLinkablePair(main('anima'), vae('flux'))).toBe(false);
    expect(isLinkablePair(main('flux2'), vae('flux'))).toBe(false);
  });

  it("use a main's variant row: Wan A14B and TI2V-5B decode with VAEs of different widths", () => {
    expect(isLinkablePair(main('wan', 'i2v_a14b'), vae('wan', 16))).toBe(true);
    expect(isLinkablePair(main('wan', 'i2v_a14b'), vae('wan', 48))).toBe(false);
    expect(isLinkablePair(main('wan', 'ti2v_5b'), vae('wan', 48))).toBe(true);
    expect(isLinkablePair(main('wan', 'ti2v_5b'), vae('wan', 16))).toBe(false);
  });

  it('let other hosts take any VAE some variant of their base decodes with, width for width', () => {
    // Wan LoRA variants (`a14b`, `5b`) are not main variants; the backend offers every Wan VAE to them.
    const wan5bLora: RelatableModel = { base: 'wan', type: 'lora', variant: '5b' };
    expect(isLinkablePair(wan5bLora, vae('wan', 48))).toBe(true);
    expect(isLinkablePair(wan5bLora, vae('wan', 16))).toBe(true);
    // Anima accepts only the 16-channel Wan VAE in every row.
    expect(isLinkablePair(model('anima'), vae('wan', 16))).toBe(true);
    expect(isLinkablePair(model('anima'), vae('wan', 48))).toBe(false);
    // An architecture without a VAE facet takes its own base only.
    expect(isLinkablePair(model('sdxl'), vae('sdxl'))).toBe(true);
    expect(isLinkablePair(model('sdxl'), vae('flux'))).toBe(false);
  });

  it('answer the same from either side of the pair', () => {
    expect(isLinkablePair(vae('flux2'), main('ernie-image'))).toBe(true);
    expect(isLinkablePair(vae('flux'), main('anima'))).toBe(false);
    expect(isLinkablePair(vae('wan', 48), main('wan', 'ti2v_5b'))).toBe(true);
  });
});

describe('without the architecture table', () => {
  it('links no VAE pair, not even one of a single base', () => {
    expect(isLinkablePair(main('sdxl'), vae('sdxl'))).toBe(false);
    expect(isLinkablePair(vae('flux'), main('z-image'))).toBe(false);
  });

  it('still links pairs the table does not decide', () => {
    expect(isLinkablePair(main('sdxl'), model('sdxl', 'lora'))).toBe(true);
    expect(isLinkablePair(model('any', 't5_encoder'), main('flux'))).toBe(true);
    expect(isLinkablePair(main('z-image'), model('flux', 'pid_decoder'))).toBe(true);
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

describe('models that serve as another type through a role', () => {
  // A directly downloaded MOSS encoder: a Text LLM record whose files also serve as the Qwen3.5 encoder.
  const roleEncoder: RelatableModel = {
    base: 'any',
    roles: [{ type: 'qwen3_5_encoder', variant: 'qwen3_5_9b' }],
    type: 'text_llm',
  };
  const plainTextLlm: RelatableModel = { base: 'any', roles: [], type: 'text_llm' };

  it('treats the model as linkable when any of its types is', () => {
    expect(isLinkableModel(roleEncoder)).toBe(true);
    expect(hasLinkableBase(roleEncoder)).toBe(true);
    expect(isLinkableModel(plainTextLlm)).toBe(false);
    expect(hasLinkableBase(plainTextLlm)).toBe(false);
    expect(isLinkableModel({ type: 'text_llm' })).toBe(false);
  });

  it('links to the hosts its role type has an allowance for, and not to others', () => {
    // The rule is by base like the native encoder's; the role variant (9B vs 4B) is not consulted.
    expect(isLinkablePair(roleEncoder, model('moss-image', 'main'))).toBe(true);
    expect(isLinkablePair(model('moss-image', 'main'), roleEncoder)).toBe(true);
    expect(isLinkablePair(roleEncoder, model('sdxl', 'main'))).toBe(false);
  });

  it('never links a plain Text LLM', () => {
    expect(isLinkablePair(plainTextLlm, model('moss-image', 'main'))).toBe(false);
    expect(isLinkablePair(model('moss-image', 'main'), plainTextLlm)).toBe(false);
  });

  it('refuses a pair that shares a single-slot type through a role', () => {
    const native: RelatableModel = { base: 'any', type: 'qwen3_5_encoder' };

    expect(isLinkablePair(roleEncoder, native)).toBe(false);
    expect(isLinkablePair(native, roleEncoder)).toBe(false);
    expect(isLinkablePair(roleEncoder, { ...roleEncoder })).toBe(false);
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
    expect(isLinkableType('control_lora')).toBe(true);
    expect(isLinkableType('flux_redux')).toBe(true);
    expect(isLinkableType('pid_decoder')).toBe(true);
  });
});
