import type { ArchitectureCapabilitiesRow } from '@features/models/architectures';

import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/models/core/architectureCapabilities';
import { architectureCapabilitiesFixture as rows } from '@features/models/core/architectureCapabilities.testing';
import { afterEach, describe, expect, it } from 'vitest';

import { getArchitectureGenerationConfig, toBaseGenerationConfig } from './architectureGenerationConfig';
import { SUPPORTED_GENERATE_BASES } from './supportedBases';

const row = (base: string, variant: string | null = null): ArchitectureCapabilitiesRow => {
  const found = rows.find((r) => r.base === base && r.variant === variant);
  if (!found) {
    throw new Error(`fixture has no row for ${base}/${variant}`);
  }
  return found;
};

afterEach(resetArchitectureCapabilities);

describe('the wire contract', () => {
  it('serves a base row for every architecture this build can generate with', () => {
    const served = new Set(rows.filter((r) => r.variant === null).map((r) => r.base));
    expect(SUPPORTED_GENERATE_BASES.filter((base) => !served.has(base))).toEqual([]);
  });
});

describe('toBaseGenerationConfig', () => {
  it('reads the guidance field for a guidance-labelled architecture', () => {
    // guidance 3.5 is distinct from cfg_scale 1.0, which disables CFG.
    expect(toBaseGenerationConfig(row('flux')).defaults.cfgScale).toBe(3.5);
    expect(toBaseGenerationConfig(row('flux', 'dev_fill')).defaults.cfgScale).toBe(30);
  });

  it('falls back to cfg_scale when a guidance-labelled architecture declares no guidance', () => {
    const schnell = toBaseGenerationConfig(row('flux', 'schnell'));
    expect(schnell.guidanceLabel).toBe('Guidance');
    expect(schnell.defaults.cfgScale).toBe(1);
    expect(schnell.defaults.steps).toBe(4);
  });

  it('reads cfg_scale for a CFG-labelled architecture', () => {
    const sdxl = toBaseGenerationConfig(row('sdxl'));
    expect(sdxl.guidanceLabel).toBe('CFG');
    expect(sdxl.defaults).toEqual({ steps: 30, cfgScale: 7, scheduler: 'euler_a' });
  });

  it('derives the optimal side from the declared area, not from width alone', () => {
    expect(toBaseGenerationConfig(row('sd-1')).dimensions.optimalSide).toBe(512);
    expect(toBaseGenerationConfig(row('sd-2')).dimensions.optimalSide).toBe(768);
    expect(toBaseGenerationConfig(row('minimax-h3')).dimensions.optimalSide).toBe(1016); // sqrt(1344*768)
  });

  it('carries the feature flags across verbatim', () => {
    const sd1 = toBaseGenerationConfig(row('sd-1'));
    expect(sd1.dimensions.grid).toBe(8);
    expect(sd1.negativePrompt).toEqual({ visible: true, usage: 'always' });
    expect(sd1.ui).toEqual({
      sdVaeOverride: true,
      colorCompensation: false,
      vaePrecision: true,
      seamless: true,
      cfgRescale: true,
      clipSkipMax: 12,
    });
  });

  it('turns an absent clip-skip ceiling into undefined rather than null', () => {
    expect(toBaseGenerationConfig(row('flux')).ui.clipSkipMax).toBeUndefined();
    expect('clipSkipMax' in toBaseGenerationConfig(row('flux')).ui).toBe(true);
  });

  it('declares ERNIE-Image scheduler choice as reaching the graph', () => {
    expect(toBaseGenerationConfig(row('ernie-image')).schedulerAppliesToGraph).toBe(true);
  });
});

describe('getArchitectureGenerationConfig', () => {
  it('hands back stable objects so policy accessors can be used in render paths', () => {
    setArchitectureCapabilities(rows);

    expect(getArchitectureGenerationConfig('sdxl')).toBe(getArchitectureGenerationConfig('sdxl'));
  });

  it('answers from the table now held after a replacement or reset', () => {
    setArchitectureCapabilities(rows);
    expect(getArchitectureGenerationConfig('sdxl')?.defaults.steps).toBe(30);

    // Same revision number as the first table after the reset, different content.
    resetArchitectureCapabilities();
    expect(getArchitectureGenerationConfig('sdxl')).toBeUndefined();
    setArchitectureCapabilities(
      rows.map((r) => (r.base === 'sdxl' && r.defaults ? { ...r, defaults: { ...r.defaults, steps: 12 } } : r))
    );

    expect(getArchitectureGenerationConfig('sdxl')?.defaults.steps).toBe(12);
  });
});
