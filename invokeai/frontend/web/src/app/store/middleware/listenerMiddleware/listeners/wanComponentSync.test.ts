import type { AnyModelConfig } from 'services/api/types';
import { describe, expect, it } from 'vitest';

import { getWanComponentUpdates } from './wanComponentSync';

const a14bCheckpoint = {
  key: 'a14b-ckpt',
  hash: 'h',
  name: 'Wan2.2 T2V A14B',
  base: 'wan',
  type: 'main',
  format: 'checkpoint',
  variant: 't2v_a14b',
  expert: 'high',
} as unknown as AnyModelConfig;

const ti2v5bCheckpoint = {
  ...a14bCheckpoint,
  key: 'ti2v-ckpt',
  name: 'Wan2.2 TI2V 5B',
  variant: 'ti2v_5b',
  expert: 'none',
} as unknown as AnyModelConfig;

const a14bDiffusers = {
  key: 'a14b-diffusers',
  hash: 'h',
  name: 'Wan2.2 T2V A14B Diffusers',
  base: 'wan',
  type: 'main',
  format: 'diffusers',
  variant: 't2v_a14b',
} as unknown as AnyModelConfig;

const ti2v5bDiffusers = {
  ...a14bDiffusers,
  key: 'ti2v-diffusers',
  name: 'Wan2.2 TI2V 5B Diffusers',
  variant: 'ti2v_5b',
} as unknown as AnyModelConfig;

/** 16-channel Wan 2.1 VAE — what A14B needs. */
const vae16 = {
  key: 'vae-16',
  hash: 'h',
  name: 'Wan 2.1 VAE',
  base: 'wan',
  type: 'vae',
  latent_channels: 16,
} as unknown as AnyModelConfig;

/** 48-channel Wan 2.2 VAE — what TI2V-5B needs. */
const vae48 = {
  ...vae16,
  key: 'vae-48',
  name: 'Wan 2.2 VAE',
  latent_channels: 48,
} as unknown as AnyModelConfig;

const encoder = {
  key: 'umt5',
  hash: 'h',
  name: 'UMT5-XXL',
  base: 'wan',
  type: 'wan_t5_encoder',
} as unknown as AnyModelConfig;

const build = (overrides: Partial<Parameters<typeof getWanComponentUpdates>[0]> = {}) =>
  getWanComponentUpdates({
    mainConfig: a14bCheckpoint,
    isSingleFileMain: true,
    selectedVae: null,
    selectedComponentSource: null,
    selectedEncoder: null,
    availableVaes: [],
    availableDiffusers: [],
    availableEncoders: [],
    ...overrides,
  });

describe('getWanComponentUpdates', () => {
  it('fills empty slots for a single-file main', () => {
    expect(
      build({ availableVaes: [vae16], availableDiffusers: [a14bDiffusers], availableEncoders: [encoder] })
    ).toEqual({ vae: vae16, componentSource: a14bDiffusers, encoder });
  });

  it('picks the VAE by latent_channels, not by install order', () => {
    // The 16-channel VAE is installed first. A TI2V-5B main needs the 48-channel one;
    // first-match would wire the wrong VAE and the loader would reject it.
    expect(build({ mainConfig: ti2v5bCheckpoint, availableVaes: [vae16, vae48] }).vae).toEqual(vae48);
    expect(build({ mainConfig: a14bCheckpoint, availableVaes: [vae48, vae16] }).vae).toEqual(vae16);
  });

  it('never falls back to a mismatched Component Source', () => {
    // Only an A14B Diffusers model is installed and the main is TI2V-5B. Wiring it would
    // load the 16-channel VAE for a 48-channel transformer.
    expect(
      build({ mainConfig: ti2v5bCheckpoint, availableDiffusers: [a14bDiffusers] }).componentSource
    ).toBeUndefined();
    expect(build({ mainConfig: ti2v5bCheckpoint, availableDiffusers: [ti2v5bDiffusers] }).componentSource).toEqual(
      ti2v5bDiffusers
    );
  });

  it('re-points slots left over from a previous main of a different variant', () => {
    // A14B was selected, auto-filling the 16-channel VAE; now TI2V-5B is selected.
    expect(
      build({
        mainConfig: ti2v5bCheckpoint,
        selectedVae: vae16,
        selectedComponentSource: a14bDiffusers,
        availableVaes: [vae16, vae48],
        availableDiffusers: [a14bDiffusers, ti2v5bDiffusers],
      })
    ).toEqual({ vae: vae48, componentSource: ti2v5bDiffusers });
  });

  it('clears an incompatible slot when nothing compatible is installed', () => {
    expect(
      build({
        mainConfig: ti2v5bCheckpoint,
        selectedVae: vae16,
        selectedComponentSource: a14bDiffusers,
        availableVaes: [vae16],
        availableDiffusers: [a14bDiffusers],
      })
    ).toEqual({ vae: null, componentSource: null });
  });

  it('leaves a compatible selection alone', () => {
    expect(
      build({
        selectedVae: vae16,
        selectedComponentSource: a14bDiffusers,
        selectedEncoder: encoder,
        availableVaes: [vae16, vae48],
        availableDiffusers: [a14bDiffusers, ti2v5bDiffusers],
        availableEncoders: [encoder],
      })
    ).toEqual({});
  });

  it('re-validates the standalone VAE for a Diffusers main too', () => {
    // The loader prefers a wired standalone VAE over the Diffusers main's own, so a
    // stale one breaks a model that is otherwise self-contained.
    expect(
      build({
        mainConfig: ti2v5bDiffusers,
        isSingleFileMain: false,
        selectedVae: vae16,
        availableVaes: [vae16, vae48],
      })
    ).toEqual({ vae: vae48 });
  });

  it('does not wire a Component Source or encoder for a Diffusers main', () => {
    expect(
      build({
        mainConfig: a14bDiffusers,
        isSingleFileMain: false,
        availableVaes: [vae16],
        availableDiffusers: [a14bDiffusers],
        availableEncoders: [encoder],
      })
    ).toEqual({ vae: vae16 });
  });

  it('treats a slot pointing at a deleted model as empty', () => {
    // The caller resolves identifiers against installed models and passes null when the
    // lookup fails, so a deleted VAE is re-filled rather than left dangling.
    expect(build({ selectedVae: null, availableVaes: [vae16] }).vae).toEqual(vae16);
  });
});
