import { describe, expect, it } from 'vitest';

import {
  clampPlaybackWindow,
  formatMediaTime,
  getFrameAtTime,
  getFrameCount,
  getFrameTime,
  getPlaybackWindowRest,
  getScrubSteps,
} from './videoTimeline';

describe('clampPlaybackWindow', () => {
  it('keeps a range that fits the media', () => {
    expect(clampPlaybackWindow({ endSeconds: 2, startSeconds: 1 }, 4)).toEqual({ endSeconds: 2, startSeconds: 1 });
  });

  it('clamps the end to the duration and the start into the media', () => {
    expect(clampPlaybackWindow({ endSeconds: 9, startSeconds: -1 }, 4)).toEqual({ endSeconds: 4, startSeconds: 0 });
  });

  it('holds nothing when the window collapses, inverts, or starts past the end', () => {
    expect(clampPlaybackWindow({ endSeconds: 1, startSeconds: 1 }, 4)).toBeNull();
    expect(clampPlaybackWindow({ endSeconds: 1, startSeconds: 2 }, 4)).toBeNull();
    expect(clampPlaybackWindow({ endSeconds: 6, startSeconds: 5 }, 4)).toBeNull();
  });

  it('keeps the requested end while the duration is unknown', () => {
    expect(clampPlaybackWindow({ endSeconds: 9, startSeconds: 1 }, 0)).toEqual({ endSeconds: 9, startSeconds: 1 });
  });
});

describe('getPlaybackWindowRest', () => {
  it("rests on the window's last frame, the one before its exclusive end", () => {
    // Frames 2 to 4 of a 10 fps clip: the end is frame 5's start.
    expect(getPlaybackWindowRest({ endSeconds: 0.5, startSeconds: 0.2 }, 10)).toBeCloseTo(0.45);
    // An end inside frame 4 still rests on frame 4.
    expect(getPlaybackWindowRest({ endSeconds: 0.47, startSeconds: 0.2 }, 10)).toBeCloseTo(0.45);
  });

  it('rests just inside the end without a frame rate, and never before the start', () => {
    expect(getPlaybackWindowRest({ endSeconds: 2, startSeconds: 1 }, null)).toBeCloseTo(1.999);
    expect(getPlaybackWindowRest({ endSeconds: 0.47, startSeconds: 0.46 }, 10)).toBe(0.46);
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
