import type { CanvasControlLayerContract, CanvasLayerContract } from '@workbench/canvas-engine/contracts';

import { createTestStubRasterBackend, type StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it } from 'vitest';

import {
  bakePixelEditSurface,
  buildMaterializedPixelLayer,
  decidePixelEdit,
  isLayerPixelEditEligible,
} from './controlPixelEdit';

const control = (overrides: Partial<CanvasControlLayerContract> = {}): CanvasControlLayerContract => ({
  adapter: { beginEndStepPct: [0.1, 0.9], controlMode: 'more_control', kind: 'controlnet', model: 'm', weight: 0.7 },
  blendMode: 'screen',
  filter: { settings: { low: 10 }, type: 'canny' },
  id: 'control',
  isEnabled: true,
  isLocked: false,
  name: 'Control',
  opacity: 0.6,
  source: { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: true,
  ...overrides,
});

describe('decidePixelEdit', () => {
  it.each([
    ['locked', control({ isLocked: true }), true, true, 'locked'],
    ['disabled', control({ isEnabled: false }), true, true, 'disabled'],
    [
      'unsupported polygon',
      control({
        source: {
          fill: '#fff',
          height: 10,
          kind: 'polygon',
          points: [],
          stroke: null,
          strokeWidth: 0,
          type: 'shape',
          width: 10,
        },
      }),
      true,
      true,
      'unsupported',
    ],
    [
      'stale image',
      control({ source: { image: { height: 10, imageName: 'image', width: 10 }, type: 'image' } }),
      true,
      false,
      'not-ready',
    ],
  ] as const)('rejects a %s control', (_scenario, layer, hasSourceContent, isCacheReady, reason) => {
    expect(decidePixelEdit({ hasSourceContent, isCacheReady, layer })).toEqual({ reason, status: 'rejected' });
  });

  it('edits an empty identity paint control directly', () => {
    expect(decidePixelEdit({ hasSourceContent: false, isCacheReady: true, layer: control() })).toEqual({
      status: 'direct',
    });
  });

  it.each([
    control({ source: { image: { height: 10, imageName: 'image', width: 10 }, type: 'image' } }),
    control({ transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 12, y: 8 } }),
  ])('materializes a ready non-direct control', (layer) => {
    expect(decidePixelEdit({ hasSourceContent: true, isCacheReady: true, layer })).toEqual({
      status: 'materialize',
    });
  });

  it('materializes a ready raster image and rejects a stale one', () => {
    const layer: CanvasLayerContract = {
      blendMode: 'normal',
      id: 'image',
      isEnabled: true,
      isLocked: false,
      name: 'Image',
      opacity: 1,
      source: { image: { height: 10, imageName: 'image', width: 10 }, type: 'image' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    };
    expect(decidePixelEdit({ hasSourceContent: true, isCacheReady: true, layer })).toEqual({
      status: 'materialize',
    });
    expect(decidePixelEdit({ hasSourceContent: true, isCacheReady: false, layer })).toEqual({
      reason: 'not-ready',
      status: 'rejected',
    });
  });
});

describe('isLayerPixelEditEligible', () => {
  const rasterPaint: CanvasLayerContract = {
    blendMode: 'normal',
    id: 'raster',
    isEnabled: true,
    isLocked: false,
    name: 'Raster',
    opacity: 1,
    source: { bitmap: null, type: 'paint' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'raster',
  };

  it.each([
    ['raster paint', rasterPaint, true],
    ['paint control', control(), true],
    [
      'image control',
      control({ source: { image: { height: 10, imageName: 'control-image', width: 10 }, type: 'image' } }),
      true,
    ],
    [
      'rectangle control',
      control({
        source: { fill: '#fff', height: 10, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 10 },
      }),
      true,
    ],
    [
      'polygon control',
      control({
        source: {
          fill: '#fff',
          height: 10,
          kind: 'polygon',
          points: [],
          stroke: null,
          strokeWidth: 0,
          type: 'shape',
          width: 10,
        },
      }),
      false,
    ],
    [
      'raster image',
      { ...rasterPaint, source: { image: { height: 10, imageName: 'i', width: 10 }, type: 'image' } },
      false,
    ],
    ['locked control', control({ isLocked: true }), false],
    ['disabled control', control({ isEnabled: false }), false],
    ['missing layer', undefined, false],
  ] as const)('returns %s eligibility for %s', (_scenario, layer, expected) => {
    expect(isLayerPixelEditEligible(layer)).toBe(expected);
  });
});

describe('buildMaterializedPixelLayer', () => {
  it('changes only source and transform for a control layer', () => {
    const before = control({
      source: { image: { height: 10, imageName: 'image', width: 10 }, type: 'image' },
      transform: { rotation: Math.PI / 2, scaleX: 2, scaleY: 1, x: 30, y: 40 },
    });
    const after = buildMaterializedPixelLayer(before, { height: 20, width: 10, x: 20, y: 40 });

    expect(after).toEqual({
      ...before,
      source: { bitmap: null, offset: { x: 20, y: 40 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    });
    expect(before.source.type).toBe('image');
  });

  it('clears raster adjustments after baking them into a paint source', () => {
    const before: CanvasLayerContract = {
      adjustments: { brightness: 0.2, contrast: -0.1, saturation: 0.3 },
      blendMode: 'normal',
      id: 'image',
      isEnabled: true,
      isLocked: false,
      name: 'Image',
      opacity: 1,
      source: { image: { height: 10, imageName: 'image', width: 10 }, type: 'image' },
      transform: { rotation: 0, scaleX: 1.5, scaleY: 1, x: 3, y: 4 },
      type: 'raster',
    };

    const after = buildMaterializedPixelLayer(before, { height: 10, width: 15, x: 3, y: 4 });

    expect(after).toEqual({
      blendMode: 'normal',
      id: 'image',
      isEnabled: true,
      isLocked: false,
      name: 'Image',
      opacity: 1,
      source: { bitmap: null, offset: { x: 3, y: 4 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    });
    expect(before).toHaveProperty('adjustments');
  });
});

describe('bakePixelEditSurface', () => {
  it('bakes offset source pixels through the complete translated, rotated, and scaled transform', () => {
    const backend = createTestStubRasterBackend();
    const source = backend.createSurface(10, 5);
    const sourceRect = { height: 5, width: 10, x: 5, y: -4 };
    const baked = bakePixelEditSurface({
      backend,
      source,
      sourceRect,
      transform: { rotation: Math.PI / 2, scaleX: 2, scaleY: 3, x: 7, y: 11 },
    });

    expect(baked.rect).toEqual({ height: 20, width: 15, x: 4, y: 21 });
    expect(baked.surface.width).toBe(15);
    expect(baked.surface.height).toBe(20);
    expect((baked.surface as StubRasterSurface).callLog).toEqual([
      { args: [1, 0, 0, 1, 0, 0], op: 'setTransform' },
      { args: [0, 0, 15, 20], op: 'clearRect' },
      { args: ['imageSmoothingEnabled', false], op: 'set' },
      {
        args: [2 * Math.cos(Math.PI / 2), 2, -3, 3 * Math.cos(Math.PI / 2), 3, -10],
        op: 'setTransform',
      },
      { args: [source.canvas, sourceRect.x, sourceRect.y], op: 'drawImage' },
      { args: [1, 0, 0, 1, 0, 0], op: 'setTransform' },
    ]);
  });
});
