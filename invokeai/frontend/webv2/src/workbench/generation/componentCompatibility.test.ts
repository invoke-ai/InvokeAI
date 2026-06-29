import { describe, expect, it } from 'vitest';

import type { ComponentModelConfig, GenerateModelConfig } from './types';

import {
  getCompatibleSelectedComponentKey,
  isAnimaQwen3Encoder,
  isAnimaVae,
  isFlux2DiffusersSourceForModel,
  isFlux2Qwen3EncoderForModel,
  isNonAnimaQwen3Encoder,
  isVaeCompatibleWithGenerateModel,
  type GenerateComponentCandidate,
} from './componentCompatibility';

const flux2Model = (variant: string): GenerateModelConfig => ({
  base: 'flux2',
  key: `flux2-${variant}`,
  name: `FLUX.2 ${variant}`,
  type: 'main',
  variant,
});

const candidate = (overrides: Partial<GenerateComponentCandidate>): GenerateComponentCandidate => ({
  base: 'any',
  type: 'qwen3_encoder',
  ...overrides,
});

describe('Generate component compatibility', () => {
  it('separates Anima Qwen3 0.6B encoders from other Qwen3 encoders', () => {
    const qwen06b = candidate({ variant: 'qwen3_06b' });
    const qwen4b = candidate({ variant: 'qwen3_4b' });

    expect(isAnimaQwen3Encoder(qwen06b)).toBe(true);
    expect(isAnimaQwen3Encoder(qwen4b)).toBe(false);
    expect(isNonAnimaQwen3Encoder(qwen06b)).toBe(false);
    expect(isNonAnimaQwen3Encoder(qwen4b)).toBe(true);
  });

  it('matches FLUX.2 Klein models to compatible Qwen3 encoder variants', () => {
    const filter = isFlux2Qwen3EncoderForModel(flux2Model('klein_9b'));

    expect(filter(candidate({ variant: 'qwen3_8b' }))).toBe(true);
    expect(filter(candidate({ variant: 'qwen3_4b' }))).toBe(false);
    expect(filter(candidate({ variant: 'qwen3_06b' }))).toBe(false);
  });

  it('matches FLUX.2 component sources by shared Qwen3 encoder variant', () => {
    const filter = isFlux2DiffusersSourceForModel(flux2Model('klein_9b'));

    expect(filter(candidate({ base: 'flux2', format: 'diffusers', type: 'main', variant: 'klein_9b_base' }))).toBe(
      true
    );
    expect(filter(candidate({ base: 'flux2', format: 'diffusers', type: 'main', variant: 'klein_4b' }))).toBe(false);
    expect(filter(candidate({ base: 'flux2', format: 'checkpoint', type: 'main', variant: 'klein_9b' }))).toBe(false);
    expect(filter(candidate({ base: 'flux2', format: 'diffusers', type: 'main' }))).toBe(false);
  });

  it('allows only backend-supported Anima VAE families', () => {
    expect(isAnimaVae(candidate({ base: 'anima', type: 'vae' }))).toBe(true);
    expect(isAnimaVae(candidate({ base: 'qwen-image', type: 'vae' }))).toBe(true);
    expect(isAnimaVae(candidate({ base: 'flux', type: 'vae' }))).toBe(true);
    expect(isAnimaVae(candidate({ base: 'sdxl', type: 'vae' }))).toBe(false);
  });

  it('centralizes generate VAE compatibility for cross-base families', () => {
    const vae = (base: string) => ({ base, key: `${base}-vae`, name: `${base} VAE`, type: 'vae' as const });

    expect(isVaeCompatibleWithGenerateModel(flux2Model('klein_9b'), vae('flux2'))).toBe(true);
    expect(isVaeCompatibleWithGenerateModel(flux2Model('klein_9b'), vae('flux'))).toBe(false);
    expect(
      isVaeCompatibleWithGenerateModel({ base: 'z-image', key: 'z-image', name: 'Z-Image', type: 'main' }, vae('flux'))
    ).toBe(true);
    expect(
      isVaeCompatibleWithGenerateModel({ base: 'anima', key: 'anima', name: 'Anima', type: 'main' }, vae('qwen-image'))
    ).toBe(true);
  });

  it('hides a stale selected component when it no longer passes the picker filter', () => {
    const staleVae: ComponentModelConfig = { base: 'sdxl', key: 'sdxl-vae', name: 'SDXL VAE', type: 'vae' };

    expect(getCompatibleSelectedComponentKey(staleVae, isAnimaVae)).toBeNull();
  });
});
