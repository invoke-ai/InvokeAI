import type * as ColorDistance from '@workbench/canvas-engine/render/fill/colorDistance';

import { createGradientSample } from '@workbench/canvas-engine/render/fill/fillImages.testing';
import { buildFillLevelMap } from '@workbench/canvas-engine/render/fill/fillLevelMap';
import { describe, expect, it, vi } from 'vitest';

const measured = vi.hoisted(() => ({ count: 0 }));

vi.mock('@workbench/canvas-engine/render/fill/colorDistance', async (importOriginal) => {
  const original = await importOriginal<typeof ColorDistance>();
  return {
    createSeedColorDistance: (...args: Parameters<typeof original.createSeedColorDistance>) => {
      const measure = original.createSeedColorDistance(...args);
      return (data: Uint8ClampedArray, offset: number) => {
        measured.count++;
        return measure(data, offset);
      };
    },
  };
});

describe('fill level map budget', () => {
  it('measures every pixel exactly once, contiguous or not', () => {
    const sample = createGradientSample(256, 192);
    for (const contiguous of [true, false]) {
      measured.count = 0;
      buildFillLevelMap(sample, 100, 60, contiguous);
      expect(measured.count).toBe(256 * 192);
    }
  });
});
