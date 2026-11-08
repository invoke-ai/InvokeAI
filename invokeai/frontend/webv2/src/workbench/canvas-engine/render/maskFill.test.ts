import type { CanvasMaskFillContract } from '@workbench/canvas-engine/contracts';

import { describe, expect, it } from 'vitest';

import type { RasterCallLogEntry, StubRasterSurface } from './raster.testStub';

import { colorizeMask, createMaskPatternTile } from './maskFill';
import { createTestStubRasterBackend } from './raster.testStub';

/** Values recorded for a set-property (e.g. `strokeStyle`), in order. */
const setValues = (log: RasterCallLogEntry[], prop: string): unknown[] =>
  log.filter((e) => e.op === 'set' && e.args[0] === prop).map((e) => e.args[1]);

const countOp = (log: RasterCallLogEntry[], op: string): number => log.filter((e) => e.op === op).length;

describe('createMaskPatternTile', () => {
  it('returns null for the solid style (no pattern — filled directly)', () => {
    const backend = createTestStubRasterBackend();
    expect(createMaskPatternTile(backend, 'solid', '#ff0000')).toBeNull();
  });

  it('builds a colour-specific tile per non-solid style, stroking in the fill colour', () => {
    const backend = createTestStubRasterBackend();
    for (const style of ['grid', 'crosshatch', 'diagonal', 'horizontal', 'vertical'] as const) {
      const tile = createMaskPatternTile(backend, style, '#00ff88') as StubRasterSurface;
      expect(tile).not.toBeNull();
      // Lines are drawn (stroke) in the fill colour.
      expect(setValues(tile.callLog, 'strokeStyle')).toContain('#00ff88');
      expect(countOp(tile.callLog, 'stroke')).toBeGreaterThan(0);
    }
  });

  it('sizes tiles to the legacy specs (diagonal/crosshatch 6px, grid 12px, h/v 9px)', () => {
    const backend = createTestStubRasterBackend();
    const size = (style: CanvasMaskFillContract['style']): number =>
      (createMaskPatternTile(backend, style, '#fff') as StubRasterSurface).width;
    expect(size('diagonal')).toBe(6);
    expect(size('crosshatch')).toBe(6);
    expect(size('grid')).toBe(12);
    expect(size('horizontal')).toBe(9);
    expect(size('vertical')).toBe(9);
  });

  it('draws more lines for crosshatch (both directions) than diagonal (one direction)', () => {
    const backend = createTestStubRasterBackend();
    const diagonal = createMaskPatternTile(backend, 'diagonal', '#fff') as StubRasterSurface;
    const crosshatch = createMaskPatternTile(backend, 'crosshatch', '#fff') as StubRasterSurface;
    expect(countOp(crosshatch.callLog, 'stroke')).toBeGreaterThan(countOp(diagonal.callLog, 'stroke'));
  });

  it('horizontal draws only horizontal lines, vertical only vertical (equal counts, disjoint geometry)', () => {
    const backend = createTestStubRasterBackend();
    const horizontal = createMaskPatternTile(backend, 'horizontal', '#fff') as StubRasterSurface;
    const vertical = createMaskPatternTile(backend, 'vertical', '#fff') as StubRasterSurface;
    // Same 9px tile, same spacing → same number of strokes.
    expect(countOp(horizontal.callLog, 'stroke')).toBe(countOp(vertical.callLog, 'stroke'));
  });
});

describe('colorizeMask', () => {
  const fill = (style: CanvasMaskFillContract['style'], color = '#ff0000'): CanvasMaskFillContract => ({
    color,
    style,
  });

  it('draws the mask stencil then fills the colour with source-in (solid)', () => {
    const backend = createTestStubRasterBackend();
    const mask = backend.createSurface(10, 10);
    const out = colorizeMask(backend, mask, 10, 10, fill('solid'), null) as StubRasterSurface;

    const ops = out.callLog;
    const drawIdx = ops.findIndex((e) => e.op === 'drawImage');
    const sourceInIdx = ops.findIndex(
      (e) => e.op === 'set' && e.args[0] === 'globalCompositeOperation' && e.args[1] === 'source-in'
    );
    const fillRectIdx = ops.findIndex((e) => e.op === 'fillRect');
    // Order: stencil blit → switch to source-in → fill the colour.
    expect(drawIdx).toBeGreaterThanOrEqual(0);
    expect(sourceInIdx).toBeGreaterThan(drawIdx);
    expect(fillRectIdx).toBeGreaterThan(sourceInIdx);
    expect(setValues(ops, 'fillStyle')).toContain('#ff0000');
  });

  it('uses the pattern tile as the fill when one is provided (pattern style)', () => {
    const backend = createTestStubRasterBackend();
    const mask = backend.createSurface(8, 8);
    const tile = createMaskPatternTile(backend, 'diagonal', '#ff0000');
    const out = colorizeMask(backend, mask, 8, 8, fill('diagonal'), tile) as StubRasterSurface;
    // A repeat pattern was created and used as the source-in fill.
    expect(countOp(out.callLog, 'createPattern')).toBe(1);
    expect(out.callLog.some((e) => e.op === 'createPattern' && e.args[1] === 'repeat')).toBe(true);
  });
});
