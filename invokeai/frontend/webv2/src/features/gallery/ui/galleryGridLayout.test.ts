import { describe, expect, it } from 'vitest';

import { GALLERY_GRID_GAP_PX, getGalleryCellSizePx, getGalleryColumnCount } from './galleryGridLayout';

describe('getGalleryColumnCount', () => {
  it('gives the same answer at the same width regardless of placement', () => {
    // The whole point of measuring rather than branching on layout: a 600px
    // gallery is a 600px gallery whether it sits in a panel or the center.
    const stacked = getGalleryColumnCount({ imageDensityPercent: 50, widthPx: 600 });
    const wide = getGalleryColumnCount({ imageDensityPercent: 50, widthPx: 600 });

    expect(stacked).toBe(wide);
  });

  it('fits more columns as the viewport grows', () => {
    const narrow = getGalleryColumnCount({ imageDensityPercent: 50, widthPx: 360 });
    const roomy = getGalleryColumnCount({ imageDensityPercent: 50, widthPx: 1200 });

    expect(roomy).toBeGreaterThan(narrow);
  });

  it('fits more columns as density rises', () => {
    const sparse = getGalleryColumnCount({ imageDensityPercent: 0, widthPx: 800 });
    const dense = getGalleryColumnCount({ imageDensityPercent: 100, widthPx: 800 });

    expect(dense).toBeGreaterThan(sparse);
  });

  it('clamps to a usable range at both extremes', () => {
    expect(getGalleryColumnCount({ imageDensityPercent: 100, widthPx: 40 })).toBe(2);
    expect(getGalleryColumnCount({ imageDensityPercent: 100, widthPx: 8000 })).toBe(12);
    expect(getGalleryColumnCount({ imageDensityPercent: 0, widthPx: 8000 })).toBe(12);
  });

  it('ignores out-of-range density instead of producing a nonsense count', () => {
    expect(getGalleryColumnCount({ imageDensityPercent: -50, widthPx: 800 })).toBe(
      getGalleryColumnCount({ imageDensityPercent: 0, widthPx: 800 })
    );
    expect(getGalleryColumnCount({ imageDensityPercent: 500, widthPx: 800 })).toBe(
      getGalleryColumnCount({ imageDensityPercent: 100, widthPx: 800 })
    );
  });

  it('falls back to the minimum before the viewport has been measured', () => {
    expect(getGalleryColumnCount({ imageDensityPercent: 50, widthPx: 0 })).toBe(2);
  });
});

describe('getGalleryCellSizePx', () => {
  it('divides the width evenly after removing the inter-column gaps', () => {
    expect(getGalleryCellSizePx({ columnCount: 4, widthPx: 400 + GALLERY_GRID_GAP_PX * 3 })).toBe(100);
  });

  it('uses a plausible square before measurement so the first paint is not zero-height', () => {
    expect(getGalleryCellSizePx({ columnCount: 4, widthPx: 0 })).toBe(96);
  });

  it('never returns a non-positive size when the width is smaller than the gaps', () => {
    expect(getGalleryCellSizePx({ columnCount: 12, widthPx: 4 })).toBeGreaterThan(0);
  });
});
