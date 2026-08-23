import { describe, expect, it } from 'vitest';

import {
  getVideoAspectRatioParts,
  getVideoDurationSeconds,
  invertVideoAspectRatioId,
  isValidMiniMaxH3NumFrames,
  isValidWanNumFrames,
  MINIMAX_H3_NUM_FRAMES_CHOICES,
  MINIMAX_H3_NUM_FRAMES_DEFAULT,
  resolveMiniMaxH3Canvas,
  scaleAndSnapWanDimensions,
  snapMiniMaxH3NumFrames,
  snapWanNumFrames,
  WAN_A14B_PIXEL_MULTIPLE,
  WAN_NUM_FRAMES_DEFAULT,
  WAN_NUM_FRAMES_MAX,
  WAN_TI2V_PIXEL_MULTIPLE,
} from './dimensions';

describe('scaleAndSnapWanDimensions', () => {
  it('scales the short side to the preset and snaps to the A14B grid of 16', () => {
    // 1920x1080 at 720p is an exact fit.
    expect(scaleAndSnapWanDimensions(1920, 1080, '720p', WAN_A14B_PIXEL_MULTIPLE)).toEqual({
      height: 720,
      width: 1280,
    });
    // 1920x1080 at 480p: 853.33 wide, snapped to the nearest multiple of 16.
    expect(scaleAndSnapWanDimensions(1920, 1080, '480p', WAN_A14B_PIXEL_MULTIPLE)).toEqual({
      height: 480,
      width: 848,
    });
  });

  it('works from pure aspect-ratio parts, matching real pixels of the same ratio', () => {
    const fromParts = scaleAndSnapWanDimensions(16, 9, '720p', WAN_A14B_PIXEL_MULTIPLE);
    const fromPixels = scaleAndSnapWanDimensions(3840, 2160, '720p', WAN_A14B_PIXEL_MULTIPLE);

    expect(fromParts).toEqual(fromPixels);
    expect(fromParts).toEqual({ height: 720, width: 1280 });
  });

  it('handles portrait sources (short side is the width)', () => {
    expect(scaleAndSnapWanDimensions(1080, 1920, '480p', WAN_A14B_PIXEL_MULTIPLE)).toEqual({
      height: 848,
      width: 480,
    });
  });

  it('snaps to the TI2V grid of 32 with banker’s rounding, matching the backend', () => {
    // 720 / 32 = 22.5: Python round() gives 22 (half-to-even), not 23.
    expect(scaleAndSnapWanDimensions(1920, 1080, '720p', WAN_TI2V_PIXEL_MULTIPLE)).toEqual({
      height: 704,
      width: 1280,
    });
  });

  it('returns null for degenerate inputs', () => {
    expect(scaleAndSnapWanDimensions(0, 1080, '720p', WAN_A14B_PIXEL_MULTIPLE)).toBeNull();
    expect(scaleAndSnapWanDimensions(-16, 9, '720p', WAN_A14B_PIXEL_MULTIPLE)).toBeNull();
    expect(scaleAndSnapWanDimensions(Number.NaN, 9, '720p', WAN_A14B_PIXEL_MULTIPLE)).toBeNull();
  });
});

describe('resolveMiniMaxH3Canvas', () => {
  it('gives the native square canvas for 1:1', () => {
    expect(resolveMiniMaxH3Canvas(1, 1, '768 highres')).toEqual({ height: 768, width: 768 });
  });

  it('applies the 768x1344 area cap before snapping (16:9 lands on 1344x768)', () => {
    expect(resolveMiniMaxH3Canvas(16, 9, '768 highres')).toEqual({ height: 768, width: 1344 });
    expect(resolveMiniMaxH3Canvas(9, 16, '768 highres')).toEqual({ height: 1344, width: 768 });
  });

  it('keeps sub-cap ratios at a 768 short edge', () => {
    expect(resolveMiniMaxH3Canvas(4, 3, '768 highres')).toEqual({ height: 768, width: 1024 });
    expect(resolveMiniMaxH3Canvas(3, 4, '768 highres')).toEqual({ height: 1024, width: 768 });
  });

  it('pins the long edge to 768 in lowres mode', () => {
    expect(resolveMiniMaxH3Canvas(2, 3, '768 lowres')).toEqual({ height: 768, width: 512 });
    expect(resolveMiniMaxH3Canvas(3, 2, '768 lowres')).toEqual({ height: 512, width: 768 });
    expect(resolveMiniMaxH3Canvas(1, 1, '768 lowres')).toEqual({ height: 768, width: 768 });
  });

  it('only the ratio matters, not the absolute pixels', () => {
    expect(resolveMiniMaxH3Canvas(1920, 1080, '768 highres')).toEqual(resolveMiniMaxH3Canvas(16, 9, '768 highres'));
  });

  it('rejects ratios beyond 1:4 / 4:1 and degenerate inputs', () => {
    expect(resolveMiniMaxH3Canvas(5, 1, '768 highres')).toBeNull();
    expect(resolveMiniMaxH3Canvas(1, 5, '768 highres')).toBeNull();
    expect(resolveMiniMaxH3Canvas(0, 1, '768 highres')).toBeNull();
    expect(resolveMiniMaxH3Canvas(1, Number.NaN, '768 highres')).toBeNull();
    // The bounds themselves are accepted.
    expect(resolveMiniMaxH3Canvas(4, 1, '768 highres')).not.toBeNull();
    expect(resolveMiniMaxH3Canvas(1, 4, '768 highres')).not.toBeNull();
  });
});

describe('aspect ratio helpers', () => {
  it('parses parts and inverts every offered preset onto another preset', () => {
    expect(getVideoAspectRatioParts('16:9')).toEqual({ height: 9, width: 16 });
    expect(invertVideoAspectRatioId('16:9')).toBe('9:16');
    expect(invertVideoAspectRatioId('9:21')).toBe('21:9');
    expect(invertVideoAspectRatioId('1:1')).toBe('1:1');
  });
});

describe('Wan frame counts', () => {
  it('accepts only 4n + 1 counts within bounds', () => {
    expect(isValidWanNumFrames(81)).toBe(true);
    expect(isValidWanNumFrames(5)).toBe(true);
    expect(isValidWanNumFrames(80)).toBe(false);
    expect(isValidWanNumFrames(4)).toBe(false);
    expect(isValidWanNumFrames(81.5)).toBe(false);
  });

  it('snaps onto the grid and clamps to the slider bounds', () => {
    expect(snapWanNumFrames(81)).toBe(81);
    expect(snapWanNumFrames(80)).toBe(81);
    expect(snapWanNumFrames(1)).toBe(5);
    expect(snapWanNumFrames(10_000)).toBe(WAN_NUM_FRAMES_MAX);
    expect(snapWanNumFrames(Number.NaN)).toBe(WAN_NUM_FRAMES_DEFAULT);
    expect(isValidWanNumFrames(snapWanNumFrames(123.7))).toBe(true);
  });
});

describe('MiniMax H3 frame counts', () => {
  it('mirrors the backend 17n + 5 grid from 90 to 345, without the still block', () => {
    expect(MINIMAX_H3_NUM_FRAMES_CHOICES[0]).toBe(90);
    expect(MINIMAX_H3_NUM_FRAMES_CHOICES[MINIMAX_H3_NUM_FRAMES_CHOICES.length - 1]).toBe(345);
    expect(MINIMAX_H3_NUM_FRAMES_CHOICES).not.toContain(5);

    for (const choice of MINIMAX_H3_NUM_FRAMES_CHOICES) {
      expect(choice % 17).toBe(5);
    }

    expect(MINIMAX_H3_NUM_FRAMES_CHOICES).toContain(MINIMAX_H3_NUM_FRAMES_DEFAULT);
  });

  it('validates and snaps onto the choice list', () => {
    expect(isValidMiniMaxH3NumFrames(124)).toBe(true);
    expect(isValidMiniMaxH3NumFrames(120)).toBe(false);
    expect(snapMiniMaxH3NumFrames(124)).toBe(124);
    expect(snapMiniMaxH3NumFrames(100)).toBe(107);
    expect(snapMiniMaxH3NumFrames(0)).toBe(90);
    expect(snapMiniMaxH3NumFrames(10_000)).toBe(345);
    expect(snapMiniMaxH3NumFrames(Number.NaN)).toBe(MINIMAX_H3_NUM_FRAMES_DEFAULT);
  });
});

describe('getVideoDurationSeconds', () => {
  it('matches the backend n / fps labeling and guards degenerate inputs', () => {
    expect(getVideoDurationSeconds(124, 24)).toBeCloseTo(5.17, 2);
    expect(getVideoDurationSeconds(81, 16)).toBeCloseTo(5.0625, 4);
    expect(getVideoDurationSeconds(81, 0)).toBeNull();
    expect(getVideoDurationSeconds(Number.NaN, 16)).toBeNull();
  });
});
