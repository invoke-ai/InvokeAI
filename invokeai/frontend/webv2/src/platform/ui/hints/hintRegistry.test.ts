import { describe, expect, it } from 'vitest';

import { FEATURE_HINTS } from './hintRegistry';

/**
 * Keeps the registry and the `hints.*` catalog in step in BOTH directions. The
 * component builds its keys by template literal, so the generic
 * `translationKeys.test.ts` scanner cannot see them — a heading that never got
 * written would otherwise surface to users as the literal string
 * `hints.tileOverlap.heading`.
 *
 * The legacy frontend drifted exactly this way: 85 feature ids, 84 catalog
 * entries, and a `hrf` key that resolved to nothing.
 */

const enModules = import.meta.glob('../../../../public/locales/en.json', { eager: true, import: 'default' });
const en = Object.values(enModules)[0] as Record<string, unknown>;
const catalog = (en.hints ?? {}) as Record<string, { heading?: unknown; paragraphs?: unknown }>;

const registryIds = Object.keys(FEATURE_HINTS).sort();

describe('feature hint registry', () => {
  it('reads a non-empty catalog', () => {
    // A broken glob would make every other assertion vacuously pass.
    expect(registryIds.length).toBeGreaterThan(20);
    expect(Object.keys(catalog).length).toBeGreaterThan(20);
  });

  it('gives every registered hint a heading and at least one paragraph', () => {
    const incomplete = registryIds.filter((id) => {
      const entry = catalog[id];

      return (
        typeof entry?.heading !== 'string' ||
        entry.heading.length === 0 ||
        !Array.isArray(entry.paragraphs) ||
        entry.paragraphs.length === 0 ||
        entry.paragraphs.some((paragraph) => typeof paragraph !== 'string' || paragraph.length === 0)
      );
    });

    expect(incomplete).toEqual([]);
  });

  it('registers every catalog entry, so no copy is unreachable', () => {
    expect(Object.keys(catalog).sort()).toEqual(registryIds);
  });

  it('points every "learn more" link at an absolute https URL', () => {
    const bad = Object.entries(FEATURE_HINTS)
      .filter(([, definition]) => 'href' in definition && !definition.href.startsWith('https://'))
      .map(([id]) => id);

    expect(bad).toEqual([]);
  });
});
