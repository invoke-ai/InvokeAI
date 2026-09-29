import { describe, expect, it } from 'vitest';

import {
  ltx2AutoDurationBounds,
  getVideoAspectRatioParts,
  ltx2ExtendJoinFitsInMemory,
  getVideoDurationSeconds,
  invertVideoAspectRatioId,
  isValidMiniMaxH3NumFrames,
  isValidWanNumFrames,
  MINIMAX_H3_NUM_FRAMES_CHOICES,
  MINIMAX_H3_NUM_FRAMES_DEFAULT,
  resolveMiniMaxH3Canvas,
  resolveMiniMaxH3ReferenceImage,
  scaleAndSnapWanDimensions,
  LTX2_NUM_FRAMES_DEFAULT,
  LTX2_NUM_FRAMES_MAX,
  LTX2_NUM_FRAMES_MIN,
  LTX2_NUM_FRAMES_STEP,
  getLtx2StageCanvases,
  isLtx2TwoStage,
  resolveLtx2Canvas,
  snapNumFramesToChoices,
  snapNumFramesToGrid,
  WAN_A14B_PIXEL_MULTIPLE,
  WAN_NUM_FRAMES_DEFAULT,
  WAN_NUM_FRAMES_MAX,
  WAN_NUM_FRAMES_MIN,
  WAN_NUM_FRAMES_STEP,
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

const WAN_GRID = {
  defaultValue: WAN_NUM_FRAMES_DEFAULT,
  max: WAN_NUM_FRAMES_MAX,
  min: WAN_NUM_FRAMES_MIN,
  step: WAN_NUM_FRAMES_STEP,
};
const snapWanFrames = (numFrames: number) => snapNumFramesToGrid(WAN_GRID, numFrames);

const H3_CHOICES = {
  choices: MINIMAX_H3_NUM_FRAMES_CHOICES,
  defaultValue: MINIMAX_H3_NUM_FRAMES_DEFAULT,
};
const snapH3Frames = (numFrames: number) => snapNumFramesToChoices(H3_CHOICES, numFrames);

describe('Wan frame counts', () => {
  it('accepts only 4n + 1 counts within bounds', () => {
    expect(isValidWanNumFrames(81)).toBe(true);
    expect(isValidWanNumFrames(5)).toBe(true);
    expect(isValidWanNumFrames(80)).toBe(false);
    expect(isValidWanNumFrames(4)).toBe(false);
    expect(isValidWanNumFrames(81.5)).toBe(false);
  });

  it('snaps onto the grid and clamps to the slider bounds', () => {
    expect(snapWanFrames(81)).toBe(81);
    expect(snapWanFrames(80)).toBe(81);
    expect(snapWanFrames(1)).toBe(5);
    expect(snapWanFrames(10_000)).toBe(WAN_NUM_FRAMES_MAX);
    expect(snapWanFrames(Number.NaN)).toBe(WAN_NUM_FRAMES_DEFAULT);
    expect(isValidWanNumFrames(snapWanFrames(123.7))).toBe(true);
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
    expect(snapH3Frames(124)).toBe(124);
    expect(snapH3Frames(100)).toBe(107);
    expect(snapH3Frames(0)).toBe(90);
    expect(snapH3Frames(10_000)).toBe(345);
    expect(snapH3Frames(Number.NaN)).toBe(MINIMAX_H3_NUM_FRAMES_DEFAULT);
  });
});

describe('LTX-2 canvas and frame grid', () => {
  it('pins the short edge and snaps both axes onto the 32 grid', () => {
    // Independently: 704 short edge at 16:9 -> 1251.6 long -> 1248 on the grid.
    expect(resolveLtx2Canvas(16, 9, '704p')).toEqual({ height: 704, width: 1248 });
    expect(resolveLtx2Canvas(9, 16, '704p')).toEqual({ height: 1248, width: 704 });
    expect(resolveLtx2Canvas(1, 1, '768p')).toEqual({ height: 768, width: 768 });
    expect(resolveLtx2Canvas(1920, 1080, '512p')).toEqual({ height: 512, width: 896 });
  });

  it('rejects degenerate inputs rather than returning a canvas', () => {
    expect(resolveLtx2Canvas(0, 100, '704p')).toBeNull();
    expect(resolveLtx2Canvas(Number.NaN, 100, '704p')).toBeNull();
  });

  it('snaps frame counts onto the 8n + 1 grid, rounding a tie up', () => {
    const grid = {
      defaultValue: LTX2_NUM_FRAMES_DEFAULT,
      max: LTX2_NUM_FRAMES_MAX,
      min: LTX2_NUM_FRAMES_MIN,
      step: LTX2_NUM_FRAMES_STEP,
    };

    expect(snapNumFramesToGrid(grid, 121)).toBe(121);
    expect(snapNumFramesToGrid(grid, 122)).toBe(121);
    // Halfway between 121 and 129: the longer clip is the better answer.
    expect(snapNumFramesToGrid(grid, 125)).toBe(129);
    expect(snapNumFramesToGrid(grid, 1)).toBe(LTX2_NUM_FRAMES_MIN);
    expect(snapNumFramesToGrid(grid, 10_000)).toBe(LTX2_NUM_FRAMES_MAX);
    expect(snapNumFramesToGrid(grid, Number.NaN)).toBe(LTX2_NUM_FRAMES_DEFAULT);
    expect((LTX2_NUM_FRAMES_MAX - 1) % LTX2_NUM_FRAMES_STEP).toBe(0);
  });
});

describe('resolveMiniMaxH3ReferenceImage', () => {
  // Cross-checked against the backend's own `resolve_reference_image_short_edge` +
  // `normalize_reference_image` (the graph encodes exactly these sizes).
  const LANDSCAPE_AREA = 1344 * 768;
  const SQUARE_AREA = 768 * 768;

  it.each([
    // source, detail, target area, normalized size, rows
    [1920, 1080, 'max', LANDSCAPE_AREA, 3648, 2048, 7296],
    [1920, 1080, 'match', LANDSCAPE_AREA, 1344, 768, 1008],
    [4032, 3024, 'max', LANDSCAPE_AREA, 2720, 2048, 5440],
    [4032, 3024, 'match', LANDSCAPE_AREA, 1184, 896, 1036],
    [3024, 4032, 'match', LANDSCAPE_AREA, 896, 1184, 1036],
    [1024, 1024, 'max', LANDSCAPE_AREA, 2048, 2048, 4096],
    [1920, 1080, 'match', SQUARE_AREA, 1024, 576, 576],
    [1024, 1024, 'match', SQUARE_AREA, 768, 768, 576],
  ] as const)(
    '%sx%s at %s detail normalizes to %sx%s',
    (width, height, detail, targetArea, expectedWidth, expectedHeight, expectedRows) => {
      expect(resolveMiniMaxH3ReferenceImage(width, height, detail, targetArea)).toEqual({
        dimensions: { height: expectedHeight, width: expectedWidth },
        rows: expectedRows,
      });
    }
  );

  it('never scales a match-detail reference above the 2048 rule', () => {
    const huge = resolveMiniMaxH3ReferenceImage(4000, 4000, 'match', 4096 * 4096);

    expect(huge?.dimensions).toEqual({ height: 2048, width: 2048 });
  });

  it('sizes a max-detail reference without a target area', () => {
    expect(resolveMiniMaxH3ReferenceImage(1920, 1080, 'max', null)?.rows).toBe(7296);
  });

  it('returns null when match detail has no area to match, or the source is degenerate', () => {
    expect(resolveMiniMaxH3ReferenceImage(1920, 1080, 'match', null)).toBeNull();
    expect(resolveMiniMaxH3ReferenceImage(1920, 1080, 'match', 0)).toBeNull();
    expect(resolveMiniMaxH3ReferenceImage(0, 1080, 'max', LANDSCAPE_AREA)).toBeNull();
    expect(resolveMiniMaxH3ReferenceImage(Number.NaN, 1080, 'max', LANDSCAPE_AREA)).toBeNull();
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

describe('LTX-2 two-stage canvases', () => {
  it('puts a two-stage canvas where halving it stays on the VAE grid', () => {
    // 32 is the VAE's grid; a two-stage preset resolves on 64 because the base pass runs at half.
    for (const [width, height] of [
      [1920, 1080],
      [1080, 1920],
      [1000, 1000],
    ]) {
      const canvas = resolveLtx2Canvas(width, height, '1024p');

      expect(canvas).not.toBeNull();
      expect(canvas!.width % 64, `${width}x${height} width`).toBe(0);
      expect(canvas!.height % 64, `${width}x${height} height`).toBe(0);
    }
  });

  it('derives the base canvas by halving, not by resolving a smaller preset', () => {
    const stages = getLtx2StageCanvases(1920, 1080, '1024p');

    expect(stages).toEqual({ base: { height: 512, width: 896 }, final: { height: 1024, width: 1792 } });
    // The x2 upscaler doubles a latent grid exactly, so this relation has to be exact.
    expect(stages!.base.width * 2).toBe(stages!.final.width);
    expect(stages!.base.height * 2).toBe(stages!.final.height);
  });

  it('gives a single-stage preset the same canvas twice, so one code path builds both', () => {
    const stages = getLtx2StageCanvases(1920, 1080, '704p');

    expect(stages!.base).toEqual(stages!.final);
    expect(stages!.final).toEqual(resolveLtx2Canvas(1920, 1080, '704p'));
  });

  it('reports which presets run two passes', () => {
    expect(isLtx2TwoStage('1024p')).toBe(true);
    expect(isLtx2TwoStage('1536p')).toBe(true);
    expect(isLtx2TwoStage('768p')).toBe(false);
  });

  it('returns null for a degenerate source, in both stages', () => {
    expect(getLtx2StageCanvases(0, 1080, '1024p')).toBeNull();
    expect(getLtx2StageCanvases(Number.NaN, 1080, '1024p')).toBeNull();
  });
});

describe('ltx2ExtendJoinFitsInMemory', () => {
  // `video_concat` buffers a crossfade at the FIRST input's native resolution and refuses over
  // 512 MiB: `width * height * 3 * (transition_frames * 2 + 13)`. Pinned at the boundary rather
  // than at comfortable sizes, because every way of getting the formula wrong -- dropping the
  // blend's working frames, counting the crossfade's two sides once, misreading the budget --
  // moves the threshold without changing the answer for a 1080p or a 4K source.
  it('matches the backend budget exactly at the last size that fits', () => {
    expect(ltx2ExtendJoinFitsInMemory(3_807_595, 1, 17)).toBe(true);
    expect(ltx2ExtendJoinFitsInMemory(3_807_596, 1, 17)).toBe(false);
  });

  it('scales with the overlap the join has to blend', () => {
    expect(ltx2ExtendJoinFitsInMemory(10_526_880, 1, 2)).toBe(true);
    expect(ltx2ExtendJoinFitsInMemory(10_526_881, 1, 2)).toBe(false);
  });

  it('accepts the resolutions a user is likely to extend from', () => {
    expect(ltx2ExtendJoinFitsInMemory(1920, 1080, 17)).toBe(true);
    expect(ltx2ExtendJoinFitsInMemory(2560, 1440, 17)).toBe(true);
    expect(ltx2ExtendJoinFitsInMemory(3840, 2160, 17)).toBe(false);
  });
});

describe('ltx2AutoDurationBounds', () => {
  it('caps the choice at the Frames value, converted at the rate the clip will play', () => {
    // The Frames value is the ceiling the run's memory was sized for.
    expect(ltx2AutoDurationBounds(24, 9, 121)).toEqual({ maxSeconds: 121 / 24, minSeconds: 1 });
    expect(ltx2AutoDurationBounds(60, 9, 241)).toEqual({ maxSeconds: 241 / 60, minSeconds: 1 });
  });

  it('keeps both ends inside the range the head was trained on', () => {
    // 481 frames at 24 fps is just past 20 s, where the head would extrapolate.
    expect(ltx2AutoDurationBounds(24, 9, 481)?.maxSeconds).toBe(20);
    // 9 frames at 24 fps is 0.375 s, under the head's 1 s floor.
    expect(ltx2AutoDurationBounds(24, 9, 121)?.minSeconds).toBe(1);
  });

  it("uses the family's shortest clip as the floor when that is longer than a second", () => {
    // At 1 fps the 9-frame minimum is 9 s.
    expect(ltx2AutoDurationBounds(1, 9, 17)).toEqual({ maxSeconds: 17, minSeconds: 9 });
  });

  it('leaves nothing to choose when the ceiling is at or under the floor', () => {
    // 17 frames at 24 fps is 0.7 s; 25 frames is 1.04 s, just over.
    expect(ltx2AutoDurationBounds(24, 9, 17)).toBeNull();
    expect(ltx2AutoDurationBounds(24, 9, 24)).toBeNull();
    expect(ltx2AutoDurationBounds(24, 9, 25)).not.toBeNull();
    expect(ltx2AutoDurationBounds(1, 9, 9)).toBeNull();
  });
});
