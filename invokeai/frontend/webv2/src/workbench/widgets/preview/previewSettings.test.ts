import { describe, expect, it } from 'vitest';

import { getNextPreviewComparisonMode } from './PreviewCompare';
import { DEFAULT_PREVIEW_COMPARISON_MODE, getPreviewComparisonMode } from './previewSettings';

describe('preview comparison settings', () => {
  it('defaults missing and invalid persisted modes to slider', () => {
    expect(getPreviewComparisonMode({})).toBe(DEFAULT_PREVIEW_COMPARISON_MODE);
    expect(getPreviewComparisonMode({ comparisonMode: 'invalid' })).toBe('slider');
  });

  it.each(['slider', 'side-by-side', 'hover'] as const)('restores the %s mode', (comparisonMode) => {
    expect(getPreviewComparisonMode({ comparisonMode })).toBe(comparisonMode);
  });

  it('cycles all three comparison modes in order', () => {
    expect(getNextPreviewComparisonMode('slider')).toBe('side-by-side');
    expect(getNextPreviewComparisonMode('side-by-side')).toBe('hover');
    expect(getNextPreviewComparisonMode('hover')).toBe('slider');
  });
});
