import type {
  CanvasInpaintMaskLayerContract,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/types';

import { describe, expect, it } from 'vitest';

import { canMergeVisibleRasters, getMergeVisibleRasterLayers } from './mergeVisible';

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
