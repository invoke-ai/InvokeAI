import { describe, expect, it } from 'vitest';

import {
  getPressureStrokeRenderBounds,
  getPressureStrokeRenderOps,
  getPressureStrokeRenderOpsFromPointIndex,
  getShouldUsePressureForBrush,
  getShouldUsePressureForEraser,
  mergeOpacityDotAlphaAtPixel,
} from './pressure';

describe('pressure helpers', () => {
  it('uses pressure for brush when width or opacity is enabled', () => {
    expect(getShouldUsePressureForBrush(false, false)).toBe(false);
    expect(getShouldUsePressureForBrush(true, false)).toBe(true);
    expect(getShouldUsePressureForBrush(false, true)).toBe(true);
  });

  it('uses pressure for eraser only when width is enabled', () => {
    expect(getShouldUsePressureForEraser(false)).toBe(false);
    expect(getShouldUsePressureForEraser(true)).toBe(true);
  });

  it('builds fixed-width opacity-sensitive render ops', () => {
    const ops = getPressureStrokeRenderOps({
      points: [10, 20, 0.25, 30, 40, 0.75],
      strokeWidth: 40,
      color: { r: 255, g: 255, b: 255, a: 0.8 },
      pressureAffectsWidth: false,
      pressureAffectsOpacity: true,
    });

    const dots = ops.filter((op) => op.type === 'dot');
    expect(ops.every((op) => op.type === 'dot')).toBe(true);
    expect(dots.length).toBeGreaterThanOrEqual(20);
    expect(dots.every((dot) => dot.radius === 20)).toBe(true);
    expect(dots.every((dot) => dot.color.a >= 0 && dot.color.a <= 1)).toBe(true);
    expect(dots[0]?.color.a).toBeGreaterThan(0.2);
    expect(dots[0]?.color.a).toBeLessThanOrEqual(0.5);
    expect(dots.at(-1)?.color.a ?? 0).toBeGreaterThan(0.25);
    expect(dots.at(-1)?.color.a ?? 0).toBeLessThan(0.7);
  });

  it('builds width-sensitive render ops when opacity is disabled', () => {
    const ops = getPressureStrokeRenderOps({
      points: [10, 20, 0.25, 30, 40, 0.75],
      strokeWidth: 40,
      color: { r: 255, g: 255, b: 255, a: 0.8 },
      pressureAffectsWidth: true,
      pressureAffectsOpacity: false,
    });

    expect(ops).toHaveLength(6);
    expect(ops[0]).toEqual({
      type: 'dot',
      x: 10,
      y: 20,
      radius: 5,
      color: { r: 255, g: 255, b: 255, a: 0.8 },
    });
    const lastOp = ops.at(-1);
    expect(lastOp?.type).toBe('dot');
    if (lastOp?.type === 'dot') {
      expect(lastOp.x).toBe(30);
      expect(lastOp.y).toBe(40);
      expect(lastOp.radius).toBe(15);
      expect(lastOp.color.r).toBe(255);
      expect(lastOp.color.g).toBe(255);
      expect(lastOp.color.b).toBe(255);
      expect(lastOp.color.a).toBe(0.8);
    }

    const segments = ops.filter((op) => op.type === 'segment');
    expect(segments).toHaveLength(4);
    expect(segments.map((segment) => segment.width)).toEqual([12.5, 17.5, 22.5, 27.5]);
    expect(segments.every((segment) => segment.color.a === 0.8)).toBe(true);
  });

  it('builds a pressure-scaled dot for single-point strokes', () => {
    const ops = getPressureStrokeRenderOps({
      points: [10, 20, 0.25],
      strokeWidth: 40,
      color: { r: 255, g: 255, b: 255, a: 0.8 },
      pressureAffectsWidth: true,
      pressureAffectsOpacity: true,
    });

    expect(ops).toEqual([
      {
        type: 'dot',
        x: 10,
        y: 20,
        radius: 5,
        color: { r: 255, g: 255, b: 255, a: 0.2 },
      },
    ]);
  });

  it('computes render bounds for opacity-pressure strokes', () => {
    const bounds = getPressureStrokeRenderBounds([
      {
        type: 'segment',
        from: { x: 10, y: 20 },
        to: { x: 30, y: 40 },
        width: 40,
        color: { r: 255, g: 255, b: 255, a: 0.4 },
      },
    ]);

    expect(bounds).toEqual({
      x: -12,
      y: -2,
      width: 64,
      height: 64,
    });
  });

  it('smooths interior pressure spikes before generating subsegments', () => {
    const ops = getPressureStrokeRenderOps({
      points: [0, 0, 0.2, 20, 0, 1, 40, 0, 0.2],
      strokeWidth: 20,
      color: { r: 255, g: 255, b: 255, a: 1 },
      pressureAffectsWidth: false,
      pressureAffectsOpacity: true,
    });

    const dots = ops.filter((op) => op.type === 'dot');
    expect(dots.length).toBeGreaterThanOrEqual(20);
    const midpoint = Math.floor(dots.length / 2);
    expect(dots[0]?.color.a).toBeGreaterThan(0.2);
    expect(dots.at(-1)?.color.a).toBeGreaterThan(0.2);
    expect(
      dots.every((dot, index) => index === 0 || index > midpoint || dot.color.a >= (dots[index - 1]?.color.a ?? 0))
    ).toBe(true);
    expect(dots.every((dot, index) => index <= midpoint || dot.color.a <= (dots[index - 1]?.color.a ?? Infinity))).toBe(
      true
    );
    const peakOpacity = Math.max(...dots.map((dot) => dot.color.a));
    expect(peakOpacity).toBeGreaterThan(0.45);
    expect(peakOpacity).toBeLessThan(0.6);
  });

  it('builds only the appended tail ops for incremental preview', () => {
    const ops = getPressureStrokeRenderOpsFromPointIndex({
      points: [0, 0, 0.2, 20, 0, 0.6, 40, 0, 1],
      strokeWidth: 40,
      color: { r: 255, g: 255, b: 255, a: 1 },
      pressureAffectsWidth: false,
      pressureAffectsOpacity: true,
      startPointIndex: 1,
    });

    const dots = ops.filter((op) => op.type === 'dot');
    expect(ops.every((op) => op.type === 'dot')).toBe(true);
    expect(dots.length).toBeGreaterThanOrEqual(20);
    expect(dots.every((dot, index) => index === 0 || dot.color.a >= (dots[index - 1]?.color.a ?? 0))).toBe(true);
    expect(dots[0]?.color.a).toBeGreaterThan(0.2);
    expect(dots[0]?.color.a).toBeLessThan(0.4);
    expect(dots.at(-1)?.color.a ?? 0).toBeGreaterThan(0.8);
    expect(dots.at(-1)?.color.a ?? 0).toBeLessThan(1);
  });

  it('composites distant opacity revisits instead of clamping them to the local pass maximum', () => {
    const samePass = mergeOpacityDotAlphaAtPixel({
      currentAlpha: 120,
      candidateAlpha: 180,
      lastStrokeDistance: 10,
      strokeDistance: 30,
      lastRadius: 20,
      radius: 20,
    });

    expect(samePass.alpha).toBe(180);

    const revisit = mergeOpacityDotAlphaAtPixel({
      currentAlpha: 120,
      candidateAlpha: 180,
      lastStrokeDistance: 10,
      strokeDistance: 80,
      lastRadius: 20,
      radius: 20,
    });

    expect(revisit.alpha).toBe(Math.round(120 + (180 * (255 - 120)) / 255));
  });
});
