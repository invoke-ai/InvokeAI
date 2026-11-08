import { describe, expect, it } from 'vitest';

import { selectStagedPreviewSource, stagedPreviewKey } from './stagingPreview';

const placement = { height: 240, opacity: 0.4, width: 320, x: 11, y: 17 };
const submitted = { height: 512, width: 512, x: -40, y: 64 };
const frame = { dataUrl: 'data:image/png;base64,AAAA', height: 64, width: 64 };
const base = {
  isVisible: true,
  progressImage: null,
  progressRect: null,
  selectedCandidate: null,
} as const;

describe('selectStagedPreviewSource', () => {
  it('returns null when nothing is staged and not generating', () => {
    expect(selectStagedPreviewSource(base)).toBeNull();
  });

  it('returns the selected candidate at its placement when visible', () => {
    expect(selectStagedPreviewSource({ ...base, selectedCandidate: { imageName: 'img-1', placement } })).toEqual({
      imageName: 'img-1',
      placement,
    });
  });

  it('returns null for a selected candidate when the preview is hidden', () => {
    expect(
      selectStagedPreviewSource({ ...base, isVisible: false, selectedCandidate: { imageName: 'img-1', placement } })
    ).toBeNull();
  });

  it('pins the selected placeholder progress frame to its submitted rect, scaled to fill it', () => {
    // The submitted rect, not the frame's native 64x64 and not wherever the bbox is now.
    expect(selectStagedPreviewSource({ ...base, progressImage: frame, progressRect: submitted })).toEqual({
      dataUrl: frame.dataUrl,
      placement: { ...submitted, opacity: 1 },
    });
  });

  it('shows the progress frame while the candidate preview is hidden', () => {
    expect(
      selectStagedPreviewSource({ ...base, isVisible: false, progressImage: frame, progressRect: submitted })
    ).toEqual({ dataUrl: frame.dataUrl, placement: { ...submitted, opacity: 1 } });
  });

  it('ignores a progress frame without a rect with area', () => {
    expect(selectStagedPreviewSource({ ...base, progressImage: frame })).toBeNull();
    expect(
      selectStagedPreviewSource({ ...base, progressImage: frame, progressRect: { ...submitted, width: 0 } })
    ).toBeNull();
  });
});

describe('stagedPreviewKey', () => {
  it('is stable per source and distinguishes the source kinds', () => {
    expect(stagedPreviewKey(null)).toBe('none');
    expect(stagedPreviewKey({ imageName: 'a', placement })).toBe(stagedPreviewKey({ imageName: 'a', placement }));
    expect(stagedPreviewKey({ imageName: 'a', placement })).not.toBe(stagedPreviewKey({ imageName: 'b', placement }));
    // A new progress frame (different data URL) yields a new key so the effect re-fires.
    expect(stagedPreviewKey({ dataUrl: 'AA', placement })).not.toBe(stagedPreviewKey({ dataUrl: 'BB', placement }));
  });

  it('includes every placement field', () => {
    const original = stagedPreviewKey({ imageName: 'a', placement });

    for (const changed of [
      { ...placement, x: 12 },
      { ...placement, y: 18 },
      { ...placement, width: 321 },
      { ...placement, height: 241 },
      { ...placement, opacity: 0.5 },
    ]) {
      expect(stagedPreviewKey({ imageName: 'a', placement: changed })).not.toBe(original);
      expect(stagedPreviewKey({ dataUrl: 'AA', placement: changed })).not.toBe(
        stagedPreviewKey({ dataUrl: 'AA', placement })
      );
    }
  });
});
