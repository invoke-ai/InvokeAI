import type { GenerateModelConfig } from '@features/generation/core/types';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import { resolveCanvasProcessingSize } from './canvasProcessingSize';

seedArchitectureCapabilities();

const sd1 = { base: 'sd-1', type: 'main' } as GenerateModelConfig;
const sdxl = { base: 'sdxl', type: 'main' } as GenerateModelConfig;

describe('resolveCanvasProcessingSize', () => {
  it('keeps the bbox on the model grid when scaling is off or unset', () => {
    expect(resolveCanvasProcessingSize(sd1, 'off', { height: 300, width: 500 }, undefined)).toEqual({
      height: 304,
      width: 504,
    });
    expect(
      resolveCanvasProcessingSize(sd1, 'off', { height: 300, width: 500 }, { height: 64, method: 'none', width: 64 })
    ).toEqual({ height: 304, width: 504 });
  });

  it('grows a small bbox along its aspect ratio until it covers the optimal area', () => {
    // SD 1.x is trained at 512²: a 256×128 region processes at 2:1 with ≥ 512² pixels.
    const size = resolveCanvasProcessingSize(
      sd1,
      'off',
      { height: 128, width: 256 },
      { height: null, method: 'auto', width: null }
    );
    expect(size.width / size.height).toBeCloseTo(2, 1);
    expect(size.width * size.height).toBeGreaterThanOrEqual(512 * 512);
    expect(size).toEqual({ height: 368, width: 728 });
  });

  it('sends a square bbox straight to the optimal square', () => {
    expect(
      resolveCanvasProcessingSize(
        sdxl,
        'off',
        { height: 200, width: 200 },
        { height: null, method: 'auto', width: null }
      )
    ).toEqual({ height: 1024, width: 1024 });
  });

  it('leaves an SDXL training resolution alone in either orientation', () => {
    const auto = { height: null, method: 'auto', width: null } as const;
    expect(resolveCanvasProcessingSize(sdxl, 'off', { height: 1216, width: 832 }, auto)).toEqual({
      height: 1216,
      width: 832,
    });
    expect(resolveCanvasProcessingSize(sdxl, 'off', { height: 832, width: 1216 }, auto)).toEqual({
      height: 832,
      width: 1216,
    });
    // The same size on another base is below its optimal area and still grows.
    expect(resolveCanvasProcessingSize(sd1, 'off', { height: 1216, width: 832 }, auto)).toEqual({
      height: 1216,
      width: 832,
    });
  });

  it('never shrinks a bbox that already covers the optimal area', () => {
    expect(
      resolveCanvasProcessingSize(
        sd1,
        'off',
        { height: 1024, width: 1536 },
        { height: null, method: 'auto', width: null }
      )
    ).toEqual({ height: 1024, width: 1536 });
  });

  it('uses the manual size snapped to the grid, falling back to the bbox side by side', () => {
    expect(
      resolveCanvasProcessingSize(
        sd1,
        'off',
        { height: 300, width: 500 },
        { height: null, method: 'manual', width: 777 }
      )
    ).toEqual({ height: 304, width: 776 });
  });

  it('snaps on the grid the model variant declares, not the one its base declares', () => {
    // 720 distinguishes 32-pixel grids from 16-pixel grids.
    const bbox = { height: 720, width: 1280 };
    const a14b = { base: 'wan', type: 'main' } as GenerateModelConfig;
    const ti2v = { base: 'wan', type: 'main', variant: 'ti2v_5b' } as GenerateModelConfig;

    expect(resolveCanvasProcessingSize(a14b, 'off', bbox, undefined)).toEqual(bbox);

    const size = resolveCanvasProcessingSize(ti2v, 'off', bbox, undefined);

    expect(size.width).toBe(1280);
    expect(size.height % 32).toBe(0);
    expect(size.height).not.toBe(720);
  });
});
