import { describe, expect, it } from 'vitest';

import { getKrea2ComponentUpdates } from './krea2ComponentSync';

const vae = { key: 'vae', hash: 'h-vae', name: 'VAE', base: 'qwen-image', type: 'vae' } as const;
const animaVae = { key: 'anima-vae', hash: 'h-anima', name: 'Anima VAE', base: 'anima', type: 'vae' } as const;
const encoder = {
  key: 'encoder',
  hash: 'h-encoder',
  name: 'Encoder',
  base: 'any',
  type: 'qwen3_vl_encoder',
} as const;

describe('getKrea2ComponentUpdates', () => {
  it('clears stale standalone components for a Diffusers model', () => {
    expect(
      getKrea2ComponentUpdates({
        format: 'diffusers',
        selectedVae: vae,
        selectedEncoder: encoder,
        availableQwenImageVaes: [vae],
        availableAnimaVaes: [animaVae],
        availableEncoders: [encoder],
      })
    ).toEqual({ vae: null, encoder: null });
  });

  it('selects installed standalone components for a non-Diffusers model', () => {
    expect(
      getKrea2ComponentUpdates({
        format: 'gguf_quantized',
        selectedVae: null,
        selectedEncoder: null,
        availableQwenImageVaes: [vae],
        availableAnimaVaes: [animaVae],
        availableEncoders: [encoder],
      })
    ).toEqual({ vae, encoder });
  });

  it('falls back to an Anima VAE and preserves explicit standalone selections', () => {
    expect(
      getKrea2ComponentUpdates({
        format: 'checkpoint',
        selectedVae: animaVae,
        selectedEncoder: encoder,
        availableQwenImageVaes: [],
        availableAnimaVaes: [animaVae],
        availableEncoders: [encoder],
      })
    ).toEqual({});
  });
});
