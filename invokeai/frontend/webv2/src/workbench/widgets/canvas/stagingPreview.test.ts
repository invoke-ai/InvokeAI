import { describe, expect, it } from 'vitest';

import { selectCanvasProgressImage, selectStagedPreviewSource, stagedPreviewKey } from './stagingPreview';

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

  it('returns null for a selected candidate when the preview is hidden', () => {
    expect(selectStagedPreviewSource({ ...base, isVisible: false, selectedImageName: 'img-1' })).toBeNull();
  });

  it('prefers the live progress frame while generating, scaled to fill the bbox', () => {
    const source = selectStagedPreviewSource({
      ...base,
      isGenerationInFlight: true,
      progressImage: { dataUrl: 'data:image/png;base64,AAAA', height: 64, width: 64 },
      selectedImageName: 'img-1',
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

describe('selectCanvasProgressImage', () => {
  const frame = (queueItemId: string) => ({
    dataUrl: 'data:image/png;base64,AAAA',
    height: 64,
    target: { itemIndex: 1, queueItemId },
    width: 64,
  });

  it('returns the frame when its queue item belongs to this canvas', () => {
    const result = selectCanvasProgressImage(frame('q-canvas'), new Set(['q-canvas']));
    expect(result).toEqual({ dataUrl: 'data:image/png;base64,AAAA', height: 64, width: 64 });
  });

  it('ignores a frame from a foreign (non-canvas / other-project) queue item', () => {
    // A generate-widget run in the same project, or another project's run, shares
    // the app-global latest-progress store — its frames must not leak onto the canvas.
    expect(selectCanvasProgressImage(frame('q-generate'), new Set(['q-canvas']))).toBeNull();
  });

  it('returns null when there is no latest frame or the frame has no target', () => {
    expect(selectCanvasProgressImage(null, new Set(['q-canvas']))).toBeNull();
    expect(selectCanvasProgressImage({ dataUrl: 'x', height: 8, width: 8 }, new Set(['q-canvas']))).toBeNull();
  });

  it('returns null when no canvas queue items are in flight', () => {
    expect(selectCanvasProgressImage(frame('q-canvas'), new Set())).toBeNull();
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
});
