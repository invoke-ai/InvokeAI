import type {
  CanvasControlLayerContract,
  CanvasInpaintMaskLayerContract,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';

import { describe, expect, it } from 'vitest';

import {
  isLayerContributing,
  isLayerEditable,
  isLayerPaintable,
  isLayerTransparencyLocked,
  isMergeableRasterLayer,
  isPixelBackedLayer,
} from './layerEligibility';

const raster = (overrides: Partial<CanvasRasterLayerContractV2> = {}): CanvasRasterLayerContractV2 =>
  ({
    id: 'r',
    isEnabled: true,
    isLocked: false,
    source: { bitmap: null, type: 'paint' },
    type: 'raster',
    ...overrides,
  }) as CanvasRasterLayerContractV2;
const mask = (overrides: Partial<CanvasInpaintMaskLayerContract> = {}): CanvasInpaintMaskLayerContract =>
  ({
    id: 'm',
    isEnabled: true,
    isLocked: false,
    mask: { bitmap: null },
    type: 'inpaint_mask',
    ...overrides,
  }) as CanvasInpaintMaskLayerContract;
const control = (overrides: Partial<CanvasControlLayerContract> = {}): CanvasControlLayerContract =>
  ({
    id: 'c',
    isEnabled: true,
    isLocked: false,
    source: { image: {}, type: 'image' },
    type: 'control',
    ...overrides,
  }) as CanvasControlLayerContract;
const textRaster = (): CanvasLayerContract =>
  raster({ source: { type: 'text' } as CanvasRasterLayerContractV2['source'] });

describe('layer eligibility', () => {
  it('separates contribution from editability', () => {
    expect(isLayerContributing(raster({ isLocked: true }))).toBe(true);
    expect(isLayerContributing(raster({ isEnabled: false }))).toBe(false);
    expect(isLayerEditable(raster())).toBe(true);
    expect(isLayerEditable(raster({ isLocked: true }))).toBe(false);
    expect(isLayerEditable(raster({ isEnabled: false }))).toBe(false);
  });

  it('allows strokes on editable raster and mask layers, never on control layers', () => {
    expect(isLayerPaintable(raster())).toBe(true);
    expect(isLayerPaintable(raster({ isTransparencyLocked: true }))).toBe(true);
    expect(isLayerPaintable(raster({ isLocked: true }))).toBe(false);
    expect(isLayerPaintable(mask())).toBe(true);
    expect(isLayerPaintable(mask({ isLocked: true }))).toBe(false);
    expect(isLayerPaintable(control())).toBe(false);
  });

  it('reports transparency lock only for raster layers', () => {
    expect(isLayerTransparencyLocked(raster())).toBe(false);
    expect(isLayerTransparencyLocked(raster({ isTransparencyLocked: true }))).toBe(true);
    expect(isLayerTransparencyLocked(mask())).toBe(false);
  });

  it('identifies pixel-backed sources', () => {
    expect(isPixelBackedLayer(raster())).toBe(true);
    expect(isPixelBackedLayer(control())).toBe(true);
    expect(isPixelBackedLayer(textRaster())).toBe(false);
    expect(isPixelBackedLayer(mask())).toBe(false);
  });

  it('merges only editable pixel-backed raster layers', () => {
    expect(isMergeableRasterLayer(raster())).toBe(true);
    expect(isMergeableRasterLayer(raster({ isLocked: true }))).toBe(false);
    expect(isMergeableRasterLayer(raster({ isEnabled: false }))).toBe(false);
    expect(isMergeableRasterLayer(textRaster())).toBe(false);
    expect(isMergeableRasterLayer(control())).toBe(false);
    expect(isMergeableRasterLayer(mask())).toBe(false);
  });
});
