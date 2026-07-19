import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/canvas-engine/contracts';

import { describe, expect, it } from 'vitest';

import { computeFitBboxToLayers, computeFitBboxToMasks, fitRectToGrid, unionRenderableBounds } from './fitBbox';

/** An axis-aligned image layer (no rotation): document bounds are `[x, y, w, h]`. */
const imageLayer = (
  id: string,
  x: number,
  y: number,
  width: number,
  height: number,
  isEnabled = true
): CanvasLayerContract =>
  ({
    blendMode: 'normal',
    id,
    isEnabled,
    isLocked: false,
    name: id,
    opacity: 1,
    source: { image: { height, imageName: `${id}-img`, width }, type: 'image' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x, y },
    type: 'raster',
  }) as CanvasLayerContract;

const maskLayer = (
  type: 'inpaint_mask' | 'regional_guidance',
  id: string,
  x: number,
  y: number,
  bitmap: { imageName: string; width: number; height: number } | null
): CanvasLayerContract =>
  ({
    blendMode: 'normal',
    id,
    isEnabled: true,
    isLocked: false,
    mask: { bitmap, fill: { color: '#e07575', style: 'diagonal' } },
    name: id,
    opacity: 1,
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x, y },
    type,
  }) as CanvasLayerContract;

const inpaintMask = (
  id: string,
  x: number,
  y: number,
  bitmap: { imageName: string; width: number; height: number } | null
): CanvasLayerContract => maskLayer('inpaint_mask', id, x, y, bitmap);

const docWith = (layers: CanvasLayerContract[]): CanvasDocumentContractV2 =>
  ({
    background: 'transparent',
    bbox: { height: 512, width: 512, x: 0, y: 0 },
    height: 1024,
    layers,
    selectedLayerId: null,
    version: 2,
    width: 1024,
  }) as CanvasDocumentContractV2;

describe('fitRectToGrid', () => {
  it('snaps the rect inward to grid multiples (top-left up, size down)', () => {
    // x=10 → ceil(10/8)*8=16 (+6); w=240 → floor((240-6)/8)*8=232.
    expect(fitRectToGrid({ height: 230, width: 240, x: 10, y: 20 }, 8)).toEqual({
      height: 224,
      width: 232,
      x: 16,
      y: 24,
    });
  });

  it('degrades to a plain integer round when the grid is <= 1 (snap-to-grid off)', () => {
    // g=1: x=ceil(3.2)=4 (+0.8), w=floor(10.9-0.8)=10; y=ceil(4.8)=5 (+0.2), h=floor(12.4-0.2)=12.
    expect(fitRectToGrid({ height: 12.4, width: 10.9, x: 3.2, y: 4.8 }, 1)).toEqual({
      height: 12,
      width: 10,
      x: 4,
      y: 5,
    });
  });
});

describe('unionRenderableBounds', () => {
  it('unions enabled, non-empty layers and skips disabled / empty ones', () => {
    const doc = docWith([
      imageLayer('a', 10, 20, 100, 100),
      imageLayer('b', 200, 200, 50, 50),
      imageLayer('c', 500, 500, 40, 40, false), // disabled → excluded
      inpaintMask('empty', 0, 0, null), // empty mask → excluded
    ]);
    expect(unionRenderableBounds(doc, () => true)).toEqual({ height: 230, width: 240, x: 10, y: 20 });
  });

  it('returns null when nothing qualifies', () => {
    expect(unionRenderableBounds(docWith([inpaintMask('empty', 0, 0, null)]), () => true)).toBeNull();
  });
});

describe('computeFitBboxToLayers', () => {
  it('fits the union of all visible content, grid-snapped', () => {
    const doc = docWith([imageLayer('a', 10, 20, 100, 100), imageLayer('b', 200, 200, 50, 50)]);
    expect(computeFitBboxToLayers(doc, 8)).toEqual({ height: 224, width: 232, x: 16, y: 24 });
  });

  it('returns null with no content (empty canvas)', () => {
    expect(computeFitBboxToLayers(docWith([]), 8)).toBeNull();
  });
});

describe('computeFitBboxToMasks', () => {
  it('fits only inpaint masks, padded then grid-snapped', () => {
    const doc = docWith([
      imageLayer('a', 0, 0, 500, 500), // ignored (not a mask)
      inpaintMask('m', 100, 100, { height: 40, imageName: 'm-img', width: 40 }),
    ]);
    // mask bounds {100,100,40,40} → pad 8 → {92,92,56,56} → grid 8 → {96,96,48,48}.
    expect(computeFitBboxToMasks(doc, 8)).toEqual({ height: 48, width: 48, x: 96, y: 96 });
  });

  it('excludes regional-guidance masks (legacy fits inpaint_mask only)', () => {
    const regional = maskLayer('regional_guidance', 'rg', 400, 400, { height: 40, imageName: 'rg-img', width: 40 });
    const doc = docWith([regional, inpaintMask('m', 100, 100, { height: 40, imageName: 'm-img', width: 40 })]);
    // Identical to the inpaint-only fit above: the regional mask contributes nothing.
    expect(computeFitBboxToMasks(doc, 8)).toEqual({ height: 48, width: 48, x: 96, y: 96 });
    // Regional masks alone ⇒ nothing to fit.
    expect(computeFitBboxToMasks(docWith([regional]), 8)).toBeNull();
  });

  it('returns null when there are no visible inpaint masks', () => {
    expect(computeFitBboxToMasks(docWith([imageLayer('a', 0, 0, 100, 100)]), 8)).toBeNull();
    expect(computeFitBboxToMasks(docWith([inpaintMask('empty', 0, 0, null)]), 8)).toBeNull();
  });
});
