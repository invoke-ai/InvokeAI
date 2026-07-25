import type {
  CanvasInpaintMaskLayerContract,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';

import { describe, expect, it } from 'vitest';

import {
  canExportRasterPsd,
  getGroupActions,
  groupVisibilityAxis,
  isGroupAllVisible,
  planGroupHiddenToggle,
  planGroupVisibilityToggle,
} from './layerGroupActions';

const raster = (source: CanvasLayerSourceContract, id = 'r'): CanvasRasterLayerContractV2 => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

const imageRef = { height: 10, imageName: 'img', width: 10 };

const inpaint = (id: string, isEnabled = true): CanvasInpaintMaskLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled,
  isLocked: false,
  mask: { bitmap: null, fill: { color: '#f00', style: 'solid' } },
  name: id,
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
});

describe('getGroupActions', () => {
  it('offers merge-visible + export-psd only for the raster group, with new + toggle everywhere', () => {
    expect(getGroupActions('raster')).toEqual(['mergeVisible', 'exportPsd', 'toggleVisibility', 'new']);
    expect(getGroupActions('control')).toEqual(['toggleVisibility', 'new']);
    expect(getGroupActions('inpaint_mask')).toEqual(['toggleVisibility', 'new']);
    expect(getGroupActions('regional_guidance')).toEqual(['toggleVisibility', 'new']);
  });

  it('renders "new" rightmost', () => {
    for (const key of ['raster', 'control', 'inpaint_mask', 'regional_guidance'] as const) {
      const actions = getGroupActions(key);
      expect(actions[actions.length - 1]).toBe('new');
    }
  });
});

describe('canExportRasterPsd', () => {
  it('is false with no exportable raster content', () => {
    expect(canExportRasterPsd([])).toBe(false);
    expect(canExportRasterPsd([inpaint('m')])).toBe(false);
    // A brand-new paint layer (no bitmap) still counts — its live cache may hold
    // unflushed strokes the export bakes.
    expect(canExportRasterPsd([raster({ bitmap: null, type: 'paint' })])).toBe(true);
  });

  it('counts image/paint/gradient/text and non-polygon shapes as content', () => {
    expect(canExportRasterPsd([raster({ image: imageRef, type: 'image' })])).toBe(true);
    expect(canExportRasterPsd([raster({ bitmap: imageRef, type: 'paint' })])).toBe(true);
    expect(
      canExportRasterPsd([
        raster({ fill: null, height: 10, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 10 }),
      ])
    ).toBe(true);
  });

  it('excludes polygon shapes (no rasterizer)', () => {
    expect(
      canExportRasterPsd([
        raster({ fill: null, height: 10, kind: 'polygon', stroke: null, strokeWidth: 0, type: 'shape', width: 10 }),
      ])
    ).toBe(false);
  });
});

describe('isGroupAllVisible', () => {
  it('is true only when every layer is enabled (empty ⇒ true)', () => {
    expect(isGroupAllVisible([])).toBe(true);
    expect(isGroupAllVisible([inpaint('a', true), inpaint('b', true)])).toBe(true);
    expect(isGroupAllVisible([inpaint('a', true), inpaint('b', false)])).toBe(false);
  });
});

describe('planGroupVisibilityToggle', () => {
  it('hides all when every layer is visible', () => {
    const layers = [inpaint('a', true), inpaint('b', true)];
    const { forward, inverse, nextVisible } = planGroupVisibilityToggle(layers);
    expect(nextVisible).toBe(false);
    expect(forward).toEqual([
      { id: 'a', isEnabled: false },
      { id: 'b', isEnabled: false },
    ]);
    // Inverse restores each layer's prior visibility verbatim (single undo entry).
    expect(inverse).toEqual([
      { id: 'a', isEnabled: true },
      { id: 'b', isEnabled: true },
    ]);
  });

  it('shows all when any layer is hidden, preserving the mixed prior state in the inverse', () => {
    const layers = [inpaint('a', true), inpaint('b', false)];
    const { forward, inverse, nextVisible } = planGroupVisibilityToggle(layers);
    expect(nextVisible).toBe(true);
    expect(forward).toEqual([
      { id: 'a', isEnabled: true },
      { id: 'b', isEnabled: true },
    ]);
    expect(inverse).toEqual([
      { id: 'a', isEnabled: true },
      { id: 'b', isEnabled: false },
    ]);
  });
});

// Merge-visible contributor selection is tested where it lives:
// `canvas-engine/document/mergeVisible.test.ts` (the selector the engine op runs).

describe('planGroupHiddenToggle', () => {
  const control = (id: string, isHidden?: boolean): CanvasLayerContract => ({
    adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
    blendMode: 'normal',
    id,
    isEnabled: true,
    isHidden,
    isLocked: false,
    name: id,
    opacity: 1,
    source: { image: { height: 10, imageName: id, width: 10 }, type: 'image' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'control',
    withTransparencyEffect: false,
  });

  it('hides every layer when all are currently shown', () => {
    const plan = planGroupHiddenToggle([control('a'), control('b')]);
    expect(plan.nextVisible).toBe(false);
    expect(plan.forward).toEqual([
      { id: 'a', isHidden: true },
      { id: 'b', isHidden: true },
    ]);
  });

  it('shows every layer when any is currently hidden', () => {
    const plan = planGroupHiddenToggle([control('a', true), control('b')]);
    expect(plan.nextVisible).toBe(true);
    expect(plan.forward).toEqual([
      { id: 'a', isHidden: false },
      { id: 'b', isHidden: false },
    ]);
  });

  it('restores each layer prior state verbatim, so undo is one entry', () => {
    const plan = planGroupHiddenToggle([control('a', true), control('b')]);
    expect(plan.inverse).toEqual([
      { id: 'a', isHidden: true },
      { id: 'b', isHidden: false },
    ]);
  });

  it('never touches enablement — hiding must not change the generated image', () => {
    const plan = planGroupHiddenToggle([control('a')]);
    expect(JSON.stringify(plan)).not.toContain('isEnabled');
  });

  it('skips layers that cannot be hidden', () => {
    const raster: CanvasLayerContract = {
      blendMode: 'normal',
      id: 'r',
      isEnabled: true,
      isLocked: false,
      name: 'r',
      opacity: 1,
      source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    };
    const plan = planGroupHiddenToggle([raster, control('a')]);
    expect(plan.forward.map((update) => update.id)).toEqual(['a']);
  });
});

describe('groupVisibilityAxis', () => {
  it('drives display visibility for the overlay groups', () => {
    expect(groupVisibilityAxis('control')).toBe('hidden');
    expect(groupVisibilityAxis('regional_guidance')).toBe('hidden');
    expect(groupVisibilityAxis('inpaint_mask')).toBe('hidden');
  });

  it('drives enablement for the raster group, which has no display axis', () => {
    expect(groupVisibilityAxis('raster')).toBe('enabled');
  });
});
