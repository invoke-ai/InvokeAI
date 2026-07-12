import { describe, expect, it } from 'vitest';

import { moveRect, resizeRectFromCorner } from './rectTransforms';

const START = { x: 100, y: 200, width: 400, height: 300 };
const OPTS = { keepAspect: false, minSize: 16 };

describe('moveRect', () => {
  it('translates by the delta without changing size', () => {
    expect(moveRect(START, 25, -50)).toEqual({ x: 125, y: 150, width: 400, height: 300 });
  });
});

describe('resizeRectFromCorner', () => {
  it('se drag grows right/down, anchoring the top-left corner', () => {
    expect(resizeRectFromCorner(START, 'se', 40, 30, OPTS)).toEqual({ x: 100, y: 200, width: 440, height: 330 });
  });

  it('nw drag anchors the bottom-right corner', () => {
    const result = resizeRectFromCorner(START, 'nw', -40, -30, OPTS);
    expect(result).toEqual({ x: 60, y: 170, width: 440, height: 330 });
    // Bottom-right corner unchanged
    expect(result.x + result.width).toBe(START.x + START.width);
    expect(result.y + result.height).toBe(START.y + START.height);
  });

  it('ne drag anchors the bottom-left corner', () => {
    const result = resizeRectFromCorner(START, 'ne', 40, -30, OPTS);
    expect(result).toEqual({ x: 100, y: 170, width: 440, height: 330 });
    expect(result.x).toBe(START.x);
    expect(result.y + result.height).toBe(START.y + START.height);
  });

  it('sw drag anchors the top-right corner', () => {
    const result = resizeRectFromCorner(START, 'sw', -40, 30, OPTS);
    expect(result).toEqual({ x: 60, y: 200, width: 440, height: 330 });
    expect(result.x + result.width).toBe(START.x + START.width);
    expect(result.y).toBe(START.y);
  });

  it('clamps each axis to minSize independently', () => {
    const result = resizeRectFromCorner(START, 'se', -1000, -1000, OPTS);
    expect(result.width).toBe(16);
    expect(result.height).toBe(16);
    // Anchored corner stays put even when clamped
    expect(result.x).toBe(START.x);
    expect(result.y).toBe(START.y);
  });

  it('nw clamp keeps the bottom-right corner anchored', () => {
    const result = resizeRectFromCorner(START, 'nw', 1000, 1000, OPTS);
    expect(result.width).toBe(16);
    expect(result.height).toBe(16);
    expect(result.x + result.width).toBe(START.x + START.width);
    expect(result.y + result.height).toBe(START.y + START.height);
  });

  it('keepAspect scales uniformly from the dominant axis', () => {
    // dx is the dominant relative change: 400 -> 600 is 1.5x
    const result = resizeRectFromCorner(START, 'se', 200, 10, { keepAspect: true, minSize: 16 });
    expect(result.width).toBeCloseTo(600);
    expect(result.height).toBeCloseTo(450);
    expect(result.width / result.height).toBeCloseTo(START.width / START.height);
  });

  it('keepAspect respects minSize on the smaller axis', () => {
    const result = resizeRectFromCorner(START, 'se', -1000, -1000, { keepAspect: true, minSize: 16 });
    expect(result.width).toBeGreaterThanOrEqual(16);
    expect(result.height).toBeGreaterThanOrEqual(16);
    expect(result.width / result.height).toBeCloseTo(START.width / START.height);
  });
});
