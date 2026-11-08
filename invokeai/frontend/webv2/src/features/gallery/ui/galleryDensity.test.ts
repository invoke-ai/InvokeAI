import { describe, expect, it } from 'vitest';

import { GALLERY_WIDE_MIN_WIDTH_PX, getGalleryLayout } from './galleryDensity';

describe('getGalleryLayout', () => {
  it('keeps side panels stacked however wide they are measured', () => {
    expect(getGalleryLayout({ region: 'left', widthPx: 2000 })).toBe('stacked');
    expect(getGalleryLayout({ region: 'right', widthPx: 2000 })).toBe('stacked');
  });

  it('switches to wide at the threshold and back below it', () => {
    expect(getGalleryLayout({ region: 'center', widthPx: GALLERY_WIDE_MIN_WIDTH_PX })).toBe('wide');
    expect(getGalleryLayout({ region: 'center', widthPx: GALLERY_WIDE_MIN_WIDTH_PX - 1 })).toBe('stacked');
    expect(getGalleryLayout({ region: 'bottom', widthPx: GALLERY_WIDE_MIN_WIDTH_PX })).toBe('wide');
    expect(getGalleryLayout({ region: 'bottom', widthPx: 320 })).toBe('stacked');
  });

  it('stays stacked before the first measurement reports a width', () => {
    expect(getGalleryLayout({ region: 'center', widthPx: 0 })).toBe('stacked');
  });
});
