import { describe, expect, it } from 'vitest';

import {
  classifySwipeIntent,
  getSwipeVelocity,
  recordSwipeSample,
  resolveSwipeRelease,
  rubberBand,
  unrubberBand,
  type SwipeSample,
} from './previewSwipe';

const WIDTH = 800;
const both = () => true;

describe('classifySwipeIntent', () => {
  it('waits inside the slop, then claims sideways travel and releases vertical travel', () => {
    expect(classifySwipeIntent(6, 6)).toBe('pending');
    expect(classifySwipeIntent(-12, 4)).toBe('swipe');
    expect(classifySwipeIntent(5, 14)).toBe('reject');
  });
});

describe('getSwipeVelocity', () => {
  const track = (points: [time: number, x: number][]): SwipeSample[] => {
    const samples: SwipeSample[] = [];

    for (const [time, x] of points) {
      recordSwipeSample(samples, { time, x });
    }

    return samples;
  };

  it('measures only the recent motion, so an early burst does not count at a slow release', () => {
    // 200px in the first 20ms, then 10px over the next 100ms.
    const samples = track([
      [0, 0],
      [20, -200],
      [70, -205],
      [120, -210],
    ]);

    expect(getSwipeVelocity(samples, 120)).toBeCloseTo(-0.1, 2);
  });

  it('reads a finger that rested before lifting as stopped', () => {
    const samples = track([
      [0, 0],
      [16, -60],
      [32, -120],
    ]);

    expect(getSwipeVelocity(samples, 32)).toBeCloseTo(-3.75, 2);
    expect(getSwipeVelocity(samples, 250)).toBe(0);
  });
});

describe('rubberBand', () => {
  it('follows small overscroll closely and never reaches a third of the width', () => {
    expect(rubberBand(20, WIDTH)).toBeGreaterThan(18);
    expect(rubberBand(10_000, WIDTH)).toBeLessThan(WIDTH * 0.3);
    expect(rubberBand(-10_000, WIDTH)).toBeGreaterThan(-WIDTH * 0.3);
  });

  it('inverts, so a caught overscroll resumes from the finger travel that produced it', () => {
    for (const travel of [-900, -40, 25, 600]) {
      expect(unrubberBand(rubberBand(travel, WIDTH), WIDTH)).toBeCloseTo(travel, 6);
    }
  });
});

describe('resolveSwipeRelease', () => {
  it('commits a short flick toward the next item, and a slow drag only past the threshold', () => {
    expect(resolveSwipeRelease({ canNavigate: both, offset: -60, velocity: -1.2, width: WIDTH })).toMatchObject({
      direction: 1,
      kind: 'commit',
    });
    expect(resolveSwipeRelease({ canNavigate: both, offset: 400, velocity: 0.05, width: WIDTH })).toMatchObject({
      direction: -1,
      kind: 'commit',
    });
    expect(resolveSwipeRelease({ canNavigate: both, offset: 200, velocity: 0.05, width: WIDTH }).kind).toBe('cancel');
  });

  it('returns a drag that is flicked back toward where it started, however far it went', () => {
    expect(resolveSwipeRelease({ canNavigate: both, offset: -600, velocity: 1.5, width: WIDTH }).kind).toBe('cancel');
  });

  it('never commits toward a missing neighbor', () => {
    const onlyPrevious = (direction: 1 | -1) => direction === -1;

    expect(resolveSwipeRelease({ canNavigate: onlyPrevious, offset: -600, velocity: -2, width: WIDTH }).kind).toBe(
      'cancel'
    );
  });

  it('finishes faster flicks sooner, within bounds', () => {
    const duration = (velocity: number) => {
      const release = resolveSwipeRelease({ canNavigate: both, offset: -100, velocity, width: WIDTH });

      return release.durationMs;
    };

    expect(duration(-4)).toBeLessThan(duration(-1));
    expect(duration(-50)).toBeGreaterThan(0);
    expect(duration(-0.5)).toBeLessThanOrEqual(320);
  });
});
