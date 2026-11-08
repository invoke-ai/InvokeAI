import { describe, expect, it } from 'vitest';

import { getPreviewDensity, PREVIEW_FULL_MIN_WIDTH_PX, PREVIEW_MINIMAL_MAX_WIDTH_PX } from './previewDensity';

describe('getPreviewDensity', () => {
  it('is minimal below the minimal threshold in any region', () => {
    expect(getPreviewDensity({ region: 'center', widthPx: PREVIEW_MINIMAL_MAX_WIDTH_PX - 1 })).toBe('minimal');
    expect(getPreviewDensity({ region: 'right', widthPx: 100 })).toBe('minimal');
  });

  it('is compact in side panels regardless of width', () => {
    expect(getPreviewDensity({ region: 'right', widthPx: 520 })).toBe('compact');
    expect(getPreviewDensity({ region: 'left', widthPx: 900 })).toBe('compact');
  });

  it('is full in the center region at or above the full threshold', () => {
    expect(getPreviewDensity({ region: 'center', widthPx: PREVIEW_FULL_MIN_WIDTH_PX })).toBe('full');
    expect(getPreviewDensity({ region: 'center', widthPx: 1200 })).toBe('full');
  });

  it('is compact in a narrow center region', () => {
    expect(getPreviewDensity({ region: 'center', widthPx: PREVIEW_FULL_MIN_WIDTH_PX - 1 })).toBe('compact');
  });
});
