import type {
  CanvasInpaintMaskLayerContract,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/types';

import { describe, expect, it } from 'vitest';

import { canMergeVisibleRasters, planMergeVisibleRuns, planNextMergeVisibleStep } from './mergeVisible';

const raster = (id: string, overrides: Partial<CanvasRasterLayerContractV2> = {}): CanvasRasterLayerContractV2 => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
  ...overrides,
});

const mask = (id: string): CanvasInpaintMaskLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: null, fill: { color: '#e07575', style: 'diagonal' } },
  name: id,
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
});

const gradientRaster = (id: string): CanvasLayerContract =>
  raster(id, {
    source: {
      angle: 0,
      height: 10,
      kind: 'linear',
      stops: [
        { color: '#000', offset: 0 },
        { color: '#fff', offset: 1 },
      ],
      type: 'gradient',
      width: 10,
    },
  });

describe('planMergeVisibleRuns', () => {
  it('scenario A: a mask between two rasters does not split the run (interleaved global order)', () => {
    // [R2(top), M, R1] — the case the round-1 adjacency planner silently no-oped.
    const layers = [raster('r2'), mask('m1'), raster('r1')];
    expect(planMergeVisibleRuns(layers)).toEqual([['r2', 'r1']]);
    expect(canMergeVisibleRasters(layers)).toBe(true);
  });

  it('scenario B: a hidden raster between two visible rasters does not split the run', () => {
    const layers = [raster('r1'), raster('r2', { isEnabled: false }), raster('r3')];
    expect(planMergeVisibleRuns(layers)).toEqual([['r1', 'r3']]);
  });

  it('scenario C: two raster clusters separated only by masks fold as ONE run (legacy parity)', () => {
    const layers = [raster('r1'), raster('r2'), mask('m1'), raster('r3'), raster('r4')];
    expect(planMergeVisibleRuns(layers)).toEqual([['r1', 'r2', 'r3', 'r4']]);
  });

  it('a visible locked raster RENDERS but cannot participate, so it splits the fold into runs', () => {
    const layers = [raster('r1'), raster('r2'), raster('locked', { isLocked: true }), raster('r3'), raster('r4')];
    expect(planMergeVisibleRuns(layers)).toEqual([
      ['r1', 'r2'],
      ['r3', 'r4'],
    ]);
  });

  it('a visible parametric raster splits the fold; runs of one are dropped', () => {
    const layers = [raster('r1'), gradientRaster('g1'), raster('r2')];
    expect(planMergeVisibleRuns(layers)).toEqual([]);
    expect(canMergeVisibleRasters(layers)).toBe(false);
  });

  it('fewer than two participants means nothing to merge', () => {
    expect(canMergeVisibleRasters([raster('r1')])).toBe(false);
    expect(canMergeVisibleRasters([raster('r1'), raster('r2', { isEnabled: false })])).toBe(false);
    expect(canMergeVisibleRasters([raster('r1'), raster('r2', { isLocked: true })])).toBe(false);
    expect(canMergeVisibleRasters([])).toBe(false);
  });
});

describe('planNextMergeVisibleStep', () => {
  it('returns a no-reorder step for a globally adjacent pair', () => {
    const step = planNextMergeVisibleStep([raster('r1'), raster('r2'), mask('m1')]);
    expect(step).toEqual({ lowerId: 'r2', orderedIds: null, upperId: 'r1' });
  });

  it('plans a reorder that moves the upper directly above the lower across interleaved layers', () => {
    // [R2, M, R1] → reorder to [M, R2, R1], then merge R2 → R1.
    const step = planNextMergeVisibleStep([raster('r2'), mask('m1'), raster('r1')]);
    expect(step).toEqual({ lowerId: 'r1', orderedIds: ['m1', 'r2', 'r1'], upperId: 'r2' });
  });

  it('the reorder slides the upper past a hidden raster, leaving everything else in place', () => {
    const step = planNextMergeVisibleStep([raster('r1'), raster('hidden', { isEnabled: false }), raster('r3')]);
    expect(step).toEqual({ lowerId: 'r3', orderedIds: ['hidden', 'r1', 'r3'], upperId: 'r1' });
  });

  it('always picks the topmost run first and returns null when nothing remains', () => {
    const step = planNextMergeVisibleStep([
      raster('r1'),
      raster('r2'),
      raster('locked', { isLocked: true }),
      raster('r3'),
      raster('r4'),
    ]);
    expect(step).toEqual({ lowerId: 'r2', orderedIds: null, upperId: 'r1' });
    expect(planNextMergeVisibleStep([raster('r1'), mask('m1')])).toBeNull();
  });
});
