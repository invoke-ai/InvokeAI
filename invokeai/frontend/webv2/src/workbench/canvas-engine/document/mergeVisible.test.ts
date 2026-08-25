import type {
  CanvasInpaintMaskLayerContract,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';

import { describe, expect, it } from 'vitest';

import {
  areSelectedRasterLayersContiguous,
  canMergeSelectedRasters,
  canMergeVisibleRasters,
  getMergeVisibleRasterLayers,
} from './mergeVisible';

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

describe('getMergeVisibleRasterLayers', () => {
  const hasContent = (id: string): boolean => id !== 'empty';

  it('returns every visible raster with content in stack order', () => {
    const layers = [
      raster('top'),
      mask('mask'),
      raster('hidden', { isEnabled: false }),
      raster('locked', { isLocked: true }),
      gradientRaster('gradient'),
      raster('empty'),
      raster('bottom'),
    ];

    expect(getMergeVisibleRasterLayers(layers, hasContent).map((layer) => layer.id)).toEqual([
      'top',
      'locked',
      'gradient',
      'bottom',
    ]);
    expect(canMergeVisibleRasters(layers, hasContent)).toBe(true);
  });

  it('requires at least two visible raster layers with content', () => {
    expect(canMergeVisibleRasters([raster('one')], hasContent)).toBe(false);
    expect(canMergeVisibleRasters([raster('one'), raster('hidden', { isEnabled: false })], hasContent)).toBe(false);
    expect(canMergeVisibleRasters([raster('one'), raster('empty')], hasContent)).toBe(false);
    expect(canMergeVisibleRasters([], hasContent)).toBe(false);
  });
});

describe('merge-selected eligibility', () => {
  const hasContent = (id: string): boolean => id !== 'empty';

  it('treats other layer groups as harmless gaps in a contiguous raster selection', () => {
    const layers = [raster('top'), mask('mask'), raster('bottom')];
    const selected = new Set(['top', 'bottom']);

    expect(areSelectedRasterLayersContiguous(layers, selected)).toBe(true);
    expect(canMergeSelectedRasters(layers, selected, hasContent)).toBe(true);
  });

  it('rejects a selection spanning an unselected raster', () => {
    const layers = [raster('top'), raster('middle'), raster('bottom')];
    const selected = new Set(['top', 'bottom']);

    expect(areSelectedRasterLayersContiguous(layers, selected)).toBe(false);
    expect(canMergeSelectedRasters(layers, selected, hasContent)).toBe(false);
  });

  it('rejects empty, locked, hidden, or non-normal raster contributors', () => {
    expect(canMergeSelectedRasters([raster('top'), raster('empty')], new Set(['top', 'empty']), hasContent)).toBe(
      false
    );
    expect(
      canMergeSelectedRasters(
        [raster('top'), raster('bottom', { isLocked: true })],
        new Set(['top', 'bottom']),
        hasContent
      )
    ).toBe(false);
    expect(
      canMergeSelectedRasters(
        [raster('top'), raster('bottom', { isEnabled: false })],
        new Set(['top', 'bottom']),
        hasContent
      )
    ).toBe(false);
    expect(
      canMergeSelectedRasters(
        [raster('top'), raster('bottom', { blendMode: 'multiply' })],
        new Set(['top', 'bottom']),
        hasContent
      )
    ).toBe(false);
  });
});
