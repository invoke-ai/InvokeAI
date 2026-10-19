import { afterEach, describe, expect, it, vi } from 'vitest';

import { detectKnownBrowserIssues } from './knownBrowserIssues';

const getWorkaroundIds = (): readonly (readonly string[])[] =>
  detectKnownBrowserIssues().map((issue) => issue.workarounds.map((workaround) => workaround.id));

afterEach(() => {
  vi.restoreAllMocks();
});

describe('known browser issues in Chromium', () => {
  it('detects nothing when the real canvas returns exact pixels', () => {
    expect(getWorkaroundIds()).toEqual([]);
  });

  it('resolves Chromium workarounds from the real navigator when readback is noised', () => {
    const readPixels = CanvasRenderingContext2D.prototype.getImageData;
    vi.spyOn(CanvasRenderingContext2D.prototype, 'getImageData').mockImplementation(function (
      this: CanvasRenderingContext2D,
      ...args: Parameters<CanvasRenderingContext2D['getImageData']>
    ) {
      const imageData = readPixels.apply(this, args);
      imageData.data[0] ^= 1;
      return imageData;
    });

    expect(getWorkaroundIds()).toEqual([['helium', 'other-browser']]);
  });
});
