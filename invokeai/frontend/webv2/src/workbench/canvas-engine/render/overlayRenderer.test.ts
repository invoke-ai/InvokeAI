import type { Mat2d, Rect } from '@workbench/canvas-engine/types';

import { identity, scale } from '@workbench/canvas-engine/math/mat2d';
import { describe, expect, it } from 'vitest';

import type { OverlayState } from './overlayRenderer';
import type { RasterCallLogEntry } from './raster.testStub';

import { MIN_GRID_SPACING_PX, renderOverlay } from './overlayRenderer';
import { createTestStubRasterBackend } from './raster.testStub';

const BBOX: Rect = { height: 40, width: 40, x: 10, y: 10 };

const findSet = (log: RasterCallLogEntry[], prop: string): unknown[] =>
  log.filter((e) => e.op === 'set' && e.args[0] === prop).map((e) => e.args[1]);

const baseState = (overrides: Partial<OverlayState> = {}): OverlayState => ({
  bbox: BBOX,
  view: identity(),
  ...overrides,
});

describe('renderOverlay', () => {
  it('clears then strokes ONLY the bbox (dashed) — no document outline (unbounded plane)', () => {
    const backend = createTestStubRasterBackend();
    const target = backend.createSurface(200, 200);

    renderOverlay(target, baseState());

    const ops = target.callLog.map((e) => e.op);
    expect(ops).toContain('clearRect');
    // The document rect is retired as a visual boundary: the bbox is the only rect stroked.
    expect(target.callLog.filter((e) => e.op === 'strokeRect')).toHaveLength(1);
    // At least one dashed setLineDash for the bbox.
    const dashArgs = target.callLog.filter((e) => e.op === 'setLineDash').map((e) => e.args[0]);
    expect(dashArgs).toContainEqual([4, 4]);
  });

  it('draws no grid when showGrid is false', () => {
    const backend = createTestStubRasterBackend();
    const target = backend.createSurface(200, 200);
    renderOverlay(target, baseState({ gridSize: 10, showGrid: false }));
    // No path building beyond stroking rects (no moveTo/lineTo for grid lines).
    expect(target.callLog.some((e) => e.op === 'moveTo')).toBe(false);
  });

  it('draws grid lines when enabled and spacing is large enough', () => {
    const backend = createTestStubRasterBackend();
    const target = backend.createSurface(200, 200);
    renderOverlay(target, baseState({ gridSize: 20, showGrid: true, view: scale(identity(), 2) }));
    expect(target.callLog.some((e) => e.op === 'moveTo')).toBe(true);
    expect(target.callLog.some((e) => e.op === 'lineTo')).toBe(true);
    expect(target.callLog.some((e) => e.op === 'stroke')).toBe(true);
  });

  it('skips the grid when zoomed out below the density threshold', () => {
    const backend = createTestStubRasterBackend();
    const target = backend.createSurface(200, 200);
    // gridSize * scale below MIN_GRID_SPACING_PX -> too dense, skip.
    const smallScale = (MIN_GRID_SPACING_PX - 1) / 10;
    const view: Mat2d = scale(identity(), smallScale);
    renderOverlay(target, baseState({ gridSize: 10, showGrid: true, view }));
    expect(target.callLog.some((e) => e.op === 'moveTo')).toBe(false);
  });

  it('draws a brush cursor ring scaled from document units', () => {
    const backend = createTestStubRasterBackend();
    const target = backend.createSurface(200, 200);
    renderOverlay(
      target,
      baseState({ cursor: { point: { x: 50, y: 50 }, radiusDoc: 10 }, view: scale(identity(), 2) })
    );

    const arc = target.callLog.find((e) => e.op === 'arc');
    expect(arc).toBeDefined();
    // center = point * 2, radius = radiusDoc * scale(2) = 20.
    expect(arc!.args.slice(0, 3)).toEqual([100, 100, 20]);
  });

  it('omits the cursor ring when cursor is null', () => {
    const backend = createTestStubRasterBackend();
    const target = backend.createSurface(200, 200);
    renderOverlay(target, baseState({ cursor: null }));
    expect(target.callLog.some((e) => e.op === 'arc')).toBe(false);
  });

  it('strokes the bbox in its own accent color (no document-outline color)', () => {
    const backend = createTestStubRasterBackend();
    const target = backend.createSurface(200, 200);
    renderOverlay(target, baseState());
    const strokeStyles = findSet(target.callLog, 'strokeStyle');
    // With the document outline gone, the bbox is the only stroked chrome here:
    // exactly one stroke color is applied.
    expect(strokeStyles).toEqual(['#3b82f6']);
  });

  it('hides the passive bbox frame when showBbox is false', () => {
    const backend = createTestStubRasterBackend();
    const target = backend.createSurface(200, 200);
    renderOverlay(target, baseState({ showBbox: false }));
    expect(target.callLog.filter((e) => e.op === 'strokeRect')).toHaveLength(0);
  });

  it('draws the bbox-overlay shade (evenodd punch-out) only when bboxOverlay is on', () => {
    const backend = createTestStubRasterBackend();
    const off = backend.createSurface(200, 200);
    renderOverlay(off, baseState());
    expect(off.callLog.some((e) => e.op === 'fill' && e.args[0] === 'evenodd')).toBe(false);

    const on = backend.createSurface(200, 200);
    renderOverlay(on, baseState({ bboxOverlay: true }));
    // One even-odd fill: the viewport rect with the bbox rect punched out.
    expect(on.callLog.filter((e) => e.op === 'fill' && e.args[0] === 'evenodd')).toHaveLength(1);
    const rects = on.callLog.filter((e) => e.op === 'rect').map((e) => e.args);
    expect(rects).toContainEqual([0, 0, 200, 200]);
    expect(rects).toContainEqual([BBOX.x, BBOX.y, BBOX.width, BBOX.height]);
  });

  it('draws rule-of-thirds guides inside the bbox when enabled', () => {
    const backend = createTestStubRasterBackend();
    const target = backend.createSurface(200, 200);
    renderOverlay(target, baseState({ ruleOfThirds: true }));
    // Four guide lines: two vertical + two horizontal moveTo/lineTo pairs.
    const moveTos = target.callLog.filter((e) => e.op === 'moveTo');
    expect(moveTos).toHaveLength(4);
  });

  it('draws the dedicated SAM mask preview before its bbox, handles, and colored points', () => {
    const backend = createTestStubRasterBackend();
    const target = backend.createSurface(200, 200);
    const mask = backend.createSurface(20, 10);
    renderOverlay(
      target,
      baseState({
        samInput: {
          bbox: { height: 30, width: 40, x: 20, y: 30 },
          excludePoints: [{ x: 50, y: 60 }],
          includePoints: [{ x: 30, y: 40 }],
        },
        samPreview: { opacity: 0.45, rect: { height: 10, width: 20, x: 20, y: 30 }, surface: mask },
        showBbox: false,
      })
    );

    const ops = target.callLog.map((entry) => entry.op);
    const previewIndex = ops.indexOf('drawImage');
    const bboxIndex = ops.indexOf('strokeRect');
    const firstPointIndex = ops.indexOf('arc');
    expect(previewIndex).toBeGreaterThan(-1);
    expect(bboxIndex).toBeGreaterThan(previewIndex);
    expect(firstPointIndex).toBeGreaterThan(bboxIndex);
    expect(findSet(target.callLog, 'fillStyle')).toEqual(expect.arrayContaining(['#22c55e', '#ef4444']));
  });

  it.each([1, 4])('keeps SAM point and handle drawing screen-constant at %sx zoom', (zoom) => {
    const backend = createTestStubRasterBackend();
    const target = backend.createSurface(500, 500);
    renderOverlay(
      target,
      baseState({
        samInput: {
          bbox: { height: 30, width: 40, x: 20, y: 30 },
          excludePoints: [],
          includePoints: [{ x: 30, y: 40 }],
        },
        showBbox: false,
        view: scale(identity(), zoom),
      })
    );

    const point = target.callLog.find((entry) => entry.op === 'arc');
    const handles = target.callLog.filter((entry) => entry.op === 'fillRect');
    expect(point?.args[2]).toBe(5);
    expect(handles).toHaveLength(8);
    expect(handles.every((entry) => entry.args[2] === 8 && entry.args[3] === 8)).toBe(true);
  });

  it('draws no SAM preview or geometry after cleanup', () => {
    const backend = createTestStubRasterBackend();
    const target = backend.createSurface(200, 200);
    renderOverlay(target, baseState({ samInput: null, samPreview: null, showBbox: false }));

    expect(target.callLog.some((entry) => entry.op === 'drawImage' || entry.op === 'arc')).toBe(false);
    expect(target.callLog.filter((entry) => entry.op === 'strokeRect')).toHaveLength(0);
  });
});
