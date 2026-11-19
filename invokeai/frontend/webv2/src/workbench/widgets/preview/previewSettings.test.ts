import { describe, expect, it } from 'vitest';

import { getNextPreviewComparisonMode } from './PreviewCompare';
import { DEFAULT_PREVIEW_COMPARISON_MODE, getPreviewComparisonMode } from './previewSettings';

describe('preview comparison settings', () => {
  it.each([
    ['defaults a missing persisted mode', {}, DEFAULT_PREVIEW_COMPARISON_MODE],
    ['defaults an invalid persisted mode to slider', { comparisonMode: 'invalid' }, 'slider'],
    ['restores the slider mode', { comparisonMode: 'slider' }, 'slider'],
    ['restores the side-by-side mode', { comparisonMode: 'side-by-side' }, 'side-by-side'],
    ['restores the hover mode', { comparisonMode: 'hover' }, 'hover'],
  ])('%s', (_, persisted, expected) => {
    expect(getPreviewComparisonMode(persisted)).toBe(expected);
  });

  it('cycles all three comparison modes in order', () => {
    expect(getNextPreviewComparisonMode('slider')).toBe('side-by-side');
    expect(getNextPreviewComparisonMode('side-by-side')).toBe('hover');
    expect(getNextPreviewComparisonMode('hover')).toBe('slider');
  });
});
