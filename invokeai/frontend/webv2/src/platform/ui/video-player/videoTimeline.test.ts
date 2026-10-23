import { describe, expect, it } from 'vitest';

import {
  clampLoopRange,
  formatMediaTime,
  getFrameAtTime,
  getFrameCount,
  getFrameTime,
  getScrubSteps,
} from './videoTimeline';

describe('clampLoopRange', () => {
  it('keeps a range that fits the media', () => {
    expect(clampLoopRange({ endSeconds: 2, startSeconds: 1 }, 4)).toEqual({ endSeconds: 2, startSeconds: 1 });
  });

  it('clamps the end to the duration and the start into the media', () => {
    expect(clampLoopRange({ endSeconds: 9, startSeconds: -1 }, 4)).toEqual({ endSeconds: 4, startSeconds: 0 });
  });

  it('loops nothing when the range collapses, inverts, or starts past the end', () => {
    expect(clampLoopRange({ endSeconds: 1, startSeconds: 1 }, 4)).toBeNull();
    expect(clampLoopRange({ endSeconds: 1, startSeconds: 2 }, 4)).toBeNull();
    expect(clampLoopRange({ endSeconds: 6, startSeconds: 5 }, 4)).toBeNull();
  });

  it('keeps the requested end while the duration is unknown', () => {
    expect(clampLoopRange({ endSeconds: 9, startSeconds: 1 }, 0)).toEqual({ endSeconds: 9, startSeconds: 1 });
  });
});

describe('frame mapping', () => {
  it('counts whole frames, tolerating a duration rounded just under the last frame', () => {
    expect(getFrameCount(1, 10)).toBe(10);
    expect(getFrameCount(0.9999999, 10)).toBe(10);
    expect(getFrameCount(1.04, 10)).toBe(10);
    expect(getFrameCount(0, 24)).toBe(1);
  });

  it('seeks to the middle of a frame, which maps back to the same frame', () => {
    expect(getFrameTime(0, 10)).toBeCloseTo(0.05);
    expect(getFrameTime(3, 24)).toBeCloseTo(3.5 / 24);

    for (const frame of [0, 1, 7, 9]) {
      expect(getFrameAtTime(getFrameTime(frame, 10), 10, 10)).toBe(frame);
    }
  });

  it('reads frame boundaries as the frame they start, and the clip end as the last frame', () => {
    expect(getFrameAtTime(0.3, 10, 10)).toBe(3);
    expect(getFrameAtTime(1, 10, 10)).toBe(9);
    expect(getFrameAtTime(-1, 10, 10)).toBe(0);
  });
});

describe('getScrubSteps', () => {
  it('steps arrow keys a hundredth of the clip, bounded for very short and very long clips', () => {
    expect(getScrubSteps(0).keyboard).toBe(0.1);
    expect(getScrubSteps(4).keyboard).toBe(0.1);
    expect(getScrubSteps(60).keyboard).toBeCloseTo(0.6);
    expect(getScrubSteps(3600).keyboard).toBe(5);
  });

  it('keeps pointer steps a thousandth of the clip, so long clips still drag finely', () => {
    expect(getScrubSteps(3600).pointer).toBeCloseTo(3.6);
    expect(getScrubSteps(600).pointer).toBeCloseTo(0.6);
    expect(getScrubSteps(0).pointer).toBe(0.001);
  });
});

describe('formatMediaTime', () => {
  it('formats minutes and hours with padded seconds', () => {
    expect(formatMediaTime(0)).toBe('0:00');
    expect(formatMediaTime(4.9)).toBe('0:04');
    expect(formatMediaTime(65)).toBe('1:05');
    expect(formatMediaTime(3725)).toBe('1:02:05');
  });

  it('reads unknown times as zero', () => {
    expect(formatMediaTime(Number.NaN)).toBe('0:00');
    expect(formatMediaTime(Number.POSITIVE_INFINITY)).toBe('0:00');
  });
});
