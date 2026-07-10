import { describe, expect, it } from 'vitest';

import { selectStagedPreviewSource, stagedPreviewKey } from './stagingPreview';

const base = {
  bboxHeight: 512,
  bboxWidth: 512,
  isGenerationInFlight: false,
  isVisible: true,
  progressImage: null,
  selectedImageName: null,
} as const;

describe('selectStagedPreviewSource', () => {
  it('returns null when nothing is staged and not generating', () => {
    expect(selectStagedPreviewSource(base)).toBeNull();
  });

  it('returns the selected candidate when visible and present', () => {
    expect(selectStagedPreviewSource({ ...base, selectedImageName: 'img-1' })).toEqual({ imageName: 'img-1' });
  });

  it('carries the selected candidate placement into the image preview', () => {
    const placement = { height: 240, opacity: 0.4, width: 320, x: 11, y: 17 };

    expect(
      selectStagedPreviewSource({
        ...base,
        selectedImageName: 'img-1',
        selectedPlacement: placement,
      })
    ).toEqual({ imageName: 'img-1', placement });
  });

  it('returns null for a selected candidate when the preview is hidden', () => {
    expect(selectStagedPreviewSource({ ...base, isVisible: false, selectedImageName: 'img-1' })).toBeNull();
  });

  it('returns the selected candidate even while another canvas slot is generating', () => {
    expect(
      selectStagedPreviewSource({
        ...base,
        isGenerationInFlight: true,
        progressImage: { dataUrl: 'data:image/png;base64,AAAA', height: 64, width: 64 },
        selectedImageName: 'img-1',
      })
    ).toEqual({ imageName: 'img-1' });
  });

  it('returns the selected placeholder progress frame while generating, scaled to fill the bbox', () => {
    const source = selectStagedPreviewSource({
      ...base,
      isGenerationInFlight: true,
      progressImage: { dataUrl: 'data:image/png;base64,AAAA', height: 64, width: 64 },
    });

    // Fills the bbox dims (512x512), not the progress frame's native 64x64.
    expect(source).toEqual({ dataUrl: 'data:image/png;base64,AAAA', height: 512, width: 512 });
  });

  it('falls back to the candidate once generation settles (no progress frame)', () => {
    expect(
      selectStagedPreviewSource({
        ...base,
        isGenerationInFlight: true,
        progressImage: null,
        selectedImageName: 'img-1',
      })
    ).toEqual({ imageName: 'img-1' });
  });

  it('ignores a progress frame when the bbox has no area', () => {
    expect(
      selectStagedPreviewSource({
        ...base,
        bboxHeight: 0,
        bboxWidth: 0,
        isGenerationInFlight: true,
        progressImage: { dataUrl: 'data:image/png;base64,AAAA', height: 64, width: 64 },
      })
    ).toBeNull();
  });
});

describe('stagedPreviewKey', () => {
  it('is stable per source and distinguishes the source kinds', () => {
    expect(stagedPreviewKey(null)).toBe('none');
    expect(stagedPreviewKey({ imageName: 'a' })).toBe('image:a');
    expect(stagedPreviewKey({ imageName: 'a' })).not.toBe(stagedPreviewKey({ imageName: 'b' }));
    // A new progress frame (different data URL) yields a new key so the effect re-fires.
    expect(stagedPreviewKey({ dataUrl: 'AA', height: 8, width: 8 })).not.toBe(
      stagedPreviewKey({ dataUrl: 'BB', height: 8, width: 8 })
    );
  });

  it('includes every image placement field', () => {
    const placement = { height: 240, opacity: 0.4, width: 320, x: 11, y: 17 };
    const original = stagedPreviewKey({ imageName: 'a', placement });

    expect(original).not.toBe(stagedPreviewKey({ imageName: 'a' }));
    expect(original).not.toBe(stagedPreviewKey({ imageName: 'a', placement: { ...placement, x: 12 } }));
    expect(original).not.toBe(stagedPreviewKey({ imageName: 'a', placement: { ...placement, y: 18 } }));
    expect(original).not.toBe(stagedPreviewKey({ imageName: 'a', placement: { ...placement, width: 321 } }));
    expect(original).not.toBe(stagedPreviewKey({ imageName: 'a', placement: { ...placement, height: 241 } }));
    expect(original).not.toBe(stagedPreviewKey({ imageName: 'a', placement: { ...placement, opacity: 0.5 } }));
  });
});
