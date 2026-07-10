import type { CanvasLayerContract, CanvasLayerSourceContract } from '@workbench/types';

import { describe, expect, it } from 'vitest';

import {
  fitThumbnailSize,
  getLayerThumbnailFallbackRenderState,
  getLayerThumbnailDisplayKey,
  nextLayerThumbnailFallbackStage,
  resolveLayerThumbnailImageRef,
} from './thumbnail';

const rasterLayer = (source: CanvasLayerSourceContract): CanvasLayerContract =>
  ({
    adjustments: { brightness: 0, contrast: 0, saturation: 0 },
    blendMode: 'normal',
    id: 'raster',
    isEnabled: true,
    isLocked: false,
    name: 'Raster',
    opacity: 1,
    source,
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'raster',
  }) as CanvasLayerContract;

const controlLayer = (source: CanvasLayerSourceContract): CanvasLayerContract =>
  ({
    adapter: {
      beginEndStepPct: [0, 1],
      controlMode: null,
      kind: 'controlnet',
      model: null,
      weight: 1,
    },
    blendMode: 'normal',
    id: 'control',
    isEnabled: true,
    isLocked: false,
    name: 'Control',
    opacity: 1,
    source,
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'control',
    withTransparencyEffect: true,
  }) as CanvasLayerContract;

const maskLayer = (type: 'inpaint_mask' | 'regional_guidance', imageName: string | null): CanvasLayerContract =>
  ({
    autoNegative: true,
    blendMode: 'normal',
    id: type,
    isEnabled: true,
    isLocked: false,
    mask: {
      bitmap: imageName ? { height: 20, imageName, width: 30 } : null,
      fill: { color: '#e07575', style: 'diagonal' },
    },
    name: type,
    negativePrompt: null,
    opacity: 1,
    positivePrompt: null,
    referenceImages: [],
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type,
  }) as CanvasLayerContract;

describe('fitThumbnailSize', () => {
  it('scales a large landscape surface to fit the box width', () => {
    expect(fitThumbnailSize(200, 100, 96)).toEqual({ height: 48, width: 96 });
  });

  it('scales a large portrait surface to fit the box height', () => {
    expect(fitThumbnailSize(100, 200, 96)).toEqual({ height: 96, width: 48 });
  });

  it('keeps a square surface square', () => {
    expect(fitThumbnailSize(300, 300, 96)).toEqual({ height: 96, width: 96 });
  });

  it('never upscales a surface smaller than the box', () => {
    expect(fitThumbnailSize(40, 20, 96)).toEqual({ height: 20, width: 40 });
  });

  it('clamps a fitted dimension to at least 1px', () => {
    expect(fitThumbnailSize(1000, 1, 96)).toEqual({ height: 1, width: 96 });
  });

  it('returns a zero size for a degenerate source or box', () => {
    expect(fitThumbnailSize(0, 100, 96)).toEqual({ height: 0, width: 0 });
    expect(fitThumbnailSize(100, 100, 0)).toEqual({ height: 0, width: 0 });
  });
});

describe('resolveLayerThumbnailImageRef', () => {
  const image = { height: 20, imageName: 'image', width: 30 };
  const bitmap = { height: 20, imageName: 'bitmap', width: 30 };

  it.each([
    ['raster image', rasterLayer({ image, type: 'image' }), image],
    ['control image', controlLayer({ image, type: 'image' }), image],
    ['raster paint', rasterLayer({ bitmap, type: 'paint' }), bitmap],
    ['control paint', controlLayer({ bitmap, type: 'paint' }), bitmap],
    ['inpaint mask', maskLayer('inpaint_mask', 'inpaint'), expect.objectContaining({ imageName: 'inpaint' })],
    [
      'regional guidance mask',
      maskLayer('regional_guidance', 'regional'),
      expect.objectContaining({ imageName: 'regional' }),
    ],
  ])('resolves the %s image reference', (_name, layer, expected) => {
    expect(resolveLayerThumbnailImageRef(layer as CanvasLayerContract)).toEqual(expected);
  });

  it.each([
    ['empty raster paint', rasterLayer({ bitmap: null, type: 'paint' })],
    ['empty control paint', controlLayer({ bitmap: null, type: 'paint' })],
    ['empty inpaint mask', maskLayer('inpaint_mask', null)],
    ['empty regional guidance mask', maskLayer('regional_guidance', null)],
    [
      'text source',
      rasterLayer({
        align: 'left',
        color: '#fff',
        content: 'text',
        fontFamily: 'Inter',
        fontSize: 16,
        fontWeight: 400,
        lineHeight: 1.2,
        type: 'text',
      }),
    ],
    [
      'shape source',
      rasterLayer({ fill: '#fff', height: 10, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 10 }),
    ],
    ['gradient source', rasterLayer({ angle: 0, kind: 'linear', stops: [], type: 'gradient' })],
  ])('returns null for an %s', (_name, layer) => {
    expect(resolveLayerThumbnailImageRef(layer as CanvasLayerContract)).toBeNull();
  });

  it('returns null instead of throwing for a malformed source', () => {
    const malformed = rasterLayer({ image, type: 'image' });
    Object.defineProperty(malformed, 'source', {
      get: () => {
        throw new Error('invalid source');
      },
    });
    expect(resolveLayerThumbnailImageRef(malformed)).toBeNull();
  });
});

describe('getLayerThumbnailDisplayKey', () => {
  it('changes only for raster adjustments, mask fill, and control transparency', () => {
    const raster = rasterLayer({ image: { height: 10, imageName: 'r', width: 10 }, type: 'image' });
    const control = controlLayer({ image: { height: 10, imageName: 'c', width: 10 }, type: 'image' });
    const mask = maskLayer('inpaint_mask', 'm');

    expect(getLayerThumbnailDisplayKey({ ...raster, name: 'renamed' })).toBe(getLayerThumbnailDisplayKey(raster));
    expect(getLayerThumbnailDisplayKey({ ...raster, opacity: 0.5 })).not.toBe(getLayerThumbnailDisplayKey(raster));
    expect(getLayerThumbnailDisplayKey({ ...control, opacity: 0.5 })).not.toBe(getLayerThumbnailDisplayKey(control));
    expect(getLayerThumbnailDisplayKey({ ...mask, opacity: 0.5 })).not.toBe(getLayerThumbnailDisplayKey(mask));
    expect(
      getLayerThumbnailDisplayKey({
        ...raster,
        adjustments: { brightness: 0.2, contrast: 0, saturation: 0 },
      } as CanvasLayerContract)
    ).not.toBe(getLayerThumbnailDisplayKey(raster));
    expect(getLayerThumbnailDisplayKey({ ...control, withTransparencyEffect: false } as CanvasLayerContract)).not.toBe(
      getLayerThumbnailDisplayKey(control)
    );
    expect(
      getLayerThumbnailDisplayKey({
        ...mask,
        mask: { ...('mask' in mask ? mask.mask : {}), fill: { color: '#00ff00', style: 'solid' } },
      } as CanvasLayerContract)
    ).not.toBe(getLayerThumbnailDisplayKey(mask));
  });
});

describe('nextLayerThumbnailFallbackStage', () => {
  it('falls back from backend thumbnail to full image, then to the retry placeholder', () => {
    expect(nextLayerThumbnailFallbackStage('thumbnail')).toBe('full');
    expect(nextLayerThumbnailFallbackStage('full')).toBe('failed');
    expect(nextLayerThumbnailFallbackStage('failed')).toBe('failed');
  });

  it('hides a failed URL fallback and its retry overlay after the engine draws successfully', () => {
    const failed = nextLayerThumbnailFallbackStage(nextLayerThumbnailFallbackStage('thumbnail'));
    expect(getLayerThumbnailFallbackRenderState(false, failed, false)).toEqual({
      showFallback: true,
      showRetry: true,
    });
    expect(getLayerThumbnailFallbackRenderState(true, failed, false)).toEqual({
      showFallback: false,
      showRetry: false,
    });
    expect(getLayerThumbnailFallbackRenderState(true, 'thumbnail', true)).toEqual({
      showFallback: false,
      showRetry: false,
    });
  });
});
