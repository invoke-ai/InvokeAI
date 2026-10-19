/**
 * __fixtures__/architectureCapabilities.json is pinned to the backend by
 * tests/backend/architectures/test_capabilities_fixture.py.
 */

import { afterEach, describe, expect, it } from 'vitest';

import fixture from './__fixtures__/architectureCapabilities.json';
import {
  type ArchitectureCapabilitiesRow,
  getArchitectureCapabilitiesRevision,
  getArchitectureCapabilityRow,
  getArchitectureFeatures,
  getArchitectureGenerationConfig,
  hasArchitectureCapabilities,
  onArchitectureCapabilitiesChanged,
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
  toBaseGenerationConfig,
} from './architectureCapabilities';
import { SUPPORTED_GENERATE_BASES } from './supportedBases';

const rows = fixture as ArchitectureCapabilitiesRow[];
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
    // Nonsquare defaults verify that optimum derives from area.
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
    // Consumers branch on presence as well as falsiness.
    expect(toBaseGenerationConfig(row('flux')).ui.clipSkipMax).toBeUndefined();
    expect('clipSkipMax' in toBaseGenerationConfig(row('flux')).ui).toBe(true);
  });

  it('declares ERNIE-Image scheduler choice as reaching the graph', () => {
    // ernie_image_denoise builds its sampler from the `scheduler` field, so the dropdown is real.
    expect(toBaseGenerationConfig(row('ernie-image')).schedulerAppliesToGraph).toBe(true);
  });
});

describe('the registry', () => {
  it('is empty until the table is pushed in', () => {
    expect(hasArchitectureCapabilities()).toBe(false);
    expect(getArchitectureGenerationConfig('sd-1')).toBeUndefined();

    setArchitectureCapabilities(rows);

    expect(hasArchitectureCapabilities()).toBe(true);
    expect(getArchitectureGenerationConfig('sd-1')?.defaults.steps).toBe(30);
  });

  it('prefers a variant row and falls back to the architecture row', () => {
    setArchitectureCapabilities(rows);

    expect(getArchitectureGenerationConfig('flux', 'schnell')?.defaults.steps).toBe(4);
    expect(getArchitectureGenerationConfig('flux', 'dev')?.defaults.steps).toBe(28);
    // A variant with no row of its own, and no variant at all, both land on the base row.
    expect(getArchitectureGenerationConfig('flux', 'made-up')?.defaults.steps).toBe(28);
    expect(getArchitectureGenerationConfig('flux')?.defaults.steps).toBe(28);
  });

  it('returns features from the architecture row regardless of variant', () => {
    setArchitectureCapabilities(rows);

    expect(getArchitectureFeatures('sd-1')?.control_kinds).toEqual(['controlnet', 't2i_adapter']);
    expect(getArchitectureFeatures('z-image')?.control_kinds).toEqual(['z_image_control']);
    expect(getArchitectureFeatures('made-up')).toBeUndefined();
  });

  it('hands back stable objects so policy accessors can be used in render paths', () => {
    setArchitectureCapabilities(rows);

    expect(getArchitectureGenerationConfig('sdxl')).toBe(getArchitectureGenerationConfig('sdxl'));
  });

  it('forgets everything on reset', () => {
    setArchitectureCapabilities(rows);
    resetArchitectureCapabilities();

    expect(hasArchitectureCapabilities()).toBe(false);
    expect(getArchitectureCapabilityRow('sd-1')).toBeUndefined();
    expect(getArchitectureFeatures('sd-1')).toBeUndefined();
  });
});

describe('publishing the table', () => {
  it('stays empty when a malformed row makes the mapping throw', () => {
    // Validate before publication so malformed rows cannot mark fallback policy loaded.
    const malformed = [...rows, { base: 'broken', variant: null } as unknown as ArchitectureCapabilitiesRow];

    expect(() => setArchitectureCapabilities(malformed)).toThrow();
    expect(hasArchitectureCapabilities()).toBe(false);
    expect(getArchitectureCapabilitiesRevision()).toBe(0);
    expect(getArchitectureGenerationConfig('sd-1')).toBeUndefined();
  });

  it('keeps the previous table intact when a later push is malformed', () => {
    setArchitectureCapabilities(rows);
    const revision = getArchitectureCapabilitiesRevision();

    expect(() =>
      setArchitectureCapabilities([{ base: 'broken', variant: null } as unknown as ArchitectureCapabilitiesRow])
    ).toThrow();
    expect(getArchitectureCapabilitiesRevision()).toBe(revision);
    expect(getArchitectureGenerationConfig('sd-1')?.defaults.steps).toBe(30);
  });

  it('notifies subscribers when the table is replaced or dropped', () => {
    // One registry subscription serves all synchronous policy readers.
    const seen: number[] = [];
    const unsubscribe = onArchitectureCapabilitiesChanged(() => seen.push(getArchitectureCapabilitiesRevision()));

    setArchitectureCapabilities(rows);
    resetArchitectureCapabilities();
    // Already empty: nothing changed, so nothing is announced.
    resetArchitectureCapabilities();
    unsubscribe();
    setArchitectureCapabilities(rows);

    expect(seen).toEqual([1, 0]);
  });
});
