/** `__fixtures__/architectureCapabilities.json` is pinned to the backend by tests/backend/architectures/test_capabilities_fixture.py. */

import { afterEach, describe, expect, it } from 'vitest';

import {
  architectureAcceptsVae,
  type ArchitectureCapabilitiesRow,
  getArchitectureCapabilitiesRevision,
  getArchitectureCapabilityRow,
  getArchitectureFeatures,
  hasArchitectureCapabilities,
  onArchitectureCapabilitiesChanged,
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
  someArchitectureVariantAcceptsVae,
  withArchitectureCapabilities,
} from './architectureCapabilities';
import { architectureCapabilitiesFixture as rows } from './architectureCapabilities.testing';

afterEach(resetArchitectureCapabilities);

describe('the registry', () => {
  it('is empty until the table is pushed in', () => {
    expect(hasArchitectureCapabilities()).toBe(false);
    expect(getArchitectureCapabilityRow('sd-1')).toBeUndefined();

    setArchitectureCapabilities(rows);

    expect(hasArchitectureCapabilities()).toBe(true);
    expect(getArchitectureCapabilityRow('sd-1')?.base).toBe('sd-1');
  });

  it('prefers a variant row and falls back to the architecture row', () => {
    setArchitectureCapabilities(rows);

    expect(getArchitectureCapabilityRow('flux', 'schnell')?.variant).toBe('schnell');
    expect(getArchitectureCapabilityRow('flux', 'made-up')?.variant).toBeNull();
    expect(getArchitectureCapabilityRow('flux')?.variant).toBeNull();
  });

  it('returns features from the architecture row when no variant is given', () => {
    setArchitectureCapabilities(rows);

    expect(getArchitectureFeatures('sd-1')?.control_kinds).toEqual(['controlnet', 't2i_adapter']);
    expect(getArchitectureFeatures('z-image')?.control_kinds).toEqual(['z_image_control']);
    expect(getArchitectureFeatures('qwen-image')?.control_kinds).toEqual(['qwen_image_control']);
    expect(getArchitectureFeatures('flux2')?.control_kinds).toEqual(['flux2_control']);
    expect(getArchitectureFeatures('made-up')).toBeUndefined();
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
  const broken = { base: 'broken', variant: null } as unknown as ArchitectureCapabilitiesRow;

  it('stays empty when a row is malformed', () => {
    expect(() => setArchitectureCapabilities([...rows, broken])).toThrow();
    expect(hasArchitectureCapabilities()).toBe(false);
    expect(getArchitectureCapabilitiesRevision()).toBe(0);
    expect(getArchitectureCapabilityRow('sd-1')).toBeUndefined();
  });

  it('keeps the previous table intact when a later push is malformed', () => {
    setArchitectureCapabilities(rows);
    const revision = getArchitectureCapabilitiesRevision();

    expect(() => setArchitectureCapabilities([broken])).toThrow();
    expect(getArchitectureCapabilitiesRevision()).toBe(revision);
    expect(getArchitectureCapabilityRow('sd-1')?.base).toBe('sd-1');
  });

  it('notifies subscribers when the table is replaced or dropped', () => {
    const seen: number[] = [];
    const unsubscribe = onArchitectureCapabilitiesChanged(() => seen.push(getArchitectureCapabilitiesRevision()));

    setArchitectureCapabilities(rows);
    resetArchitectureCapabilities();
    resetArchitectureCapabilities();
    unsubscribe();
    setArchitectureCapabilities(rows);

    expect(seen).toEqual([1, 0]);
  });
});

describe('withArchitectureCapabilities', () => {
  it('holds the table only while its callback runs, even when it throws', () => {
    expect(withArchitectureCapabilities(rows, () => hasArchitectureCapabilities())).toBe(true);
    expect(hasArchitectureCapabilities()).toBe(false);

    expect(() =>
      withArchitectureCapabilities(rows, () => {
        throw new Error('compile failed');
      })
    ).toThrow('compile failed');
    expect(hasArchitectureCapabilities()).toBe(false);
  });
});

describe('architectureAcceptsVae', () => {
  const ALL_BASES = [...new Set(rows.map((row) => row.base))];
  const host = (row: ArchitectureCapabilitiesRow) => ({ base: row.base, variant: row.variant ?? undefined });

  it('accepts nothing before the table has loaded', () => {
    expect(architectureAcceptsVae({ base: 'sdxl' }, { base: 'sdxl' })).toBe(false);
  });

  describe('with the served table', () => {
    // Expectations come from the served rows, so the rule is checked against what the backend declares.
    const seeded = <T>(run: () => T): T => withArchitectureCapabilities(rows, run);

    it('accepts every VAE base the backend declares for each row', () => {
      const declared = rows.filter((row) => row.vae);
      expect(declared.length).toBeGreaterThan(0);

      seeded(() => {
        for (const row of declared) {
          for (const accepted of row.vae!.accepted) {
            expect(
              architectureAcceptsVae(host(row), { base: accepted.base, latent_channels: accepted.latent_channels }),
              `${row.base}/${row.variant} should accept a VAE registered under ${accepted.base}`
            ).toBe(true);
          }
        }
      });
    });

    it('refuses every VAE base, and every other latent width, the backend does not declare', () => {
      seeded(() => {
        for (const row of rows) {
          const accepted = row.vae?.accepted ?? [{ base: row.base, latent_channels: null }];
          const acceptedBases = new Set(accepted.map((entry) => entry.base));

          for (const base of ALL_BASES.filter((candidate) => !acceptedBases.has(candidate))) {
            expect(
              architectureAcceptsVae(host(row), { base }),
              `${row.base}/${row.variant} should refuse a VAE registered under ${base}`
            ).toBe(false);
          }

          for (const entry of accepted.filter((candidate) => candidate.latent_channels !== null)) {
            const otherWidth = entry.latent_channels === 16 ? 48 : 16;
            if (accepted.some((other) => other.base === entry.base && other.latent_channels === otherWidth)) {
              continue;
            }
            expect(
              architectureAcceptsVae(host(row), { base: entry.base, latent_channels: otherWidth }),
              `${row.base}/${row.variant} should refuse a ${otherWidth}-channel ${entry.base} VAE`
            ).toBe(false);
          }
        }
      });
    });

    it('refuses an architecture the table says nothing about', () => {
      seeded(() => {
        expect(architectureAcceptsVae({ base: 'not-an-architecture' }, { base: 'not-an-architecture' })).toBe(false);
      });
    });
  });
});

describe('someArchitectureVariantAcceptsVae', () => {
  it("accepts what any row of the base accepts, keeping each row's latent width", () => {
    withArchitectureCapabilities(rows, () => {
      // Wan's own row takes the 16-channel VAE and its TI2V-5B row the 48-channel one.
      expect(someArchitectureVariantAcceptsVae('wan', { base: 'wan', latent_channels: 16 })).toBe(true);
      expect(someArchitectureVariantAcceptsVae('wan', { base: 'wan', latent_channels: 48 })).toBe(true);
      // No Anima row takes the 48-channel one.
      expect(someArchitectureVariantAcceptsVae('anima', { base: 'wan', latent_channels: 48 })).toBe(false);
      expect(someArchitectureVariantAcceptsVae('not-an-architecture', { base: 'not-an-architecture' })).toBe(false);
    });
  });

  it('accepts nothing before the table has loaded', () => {
    expect(someArchitectureVariantAcceptsVae('wan', { base: 'wan', latent_channels: 16 })).toBe(false);
  });
});
