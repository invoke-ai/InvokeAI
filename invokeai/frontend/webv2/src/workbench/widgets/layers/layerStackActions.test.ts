import type { CanvasLayerContract, CanvasLayerSourceContract, CanvasNodeContract } from '@workbench/canvas-engine/api';

import { compileDocumentNodes } from '@workbench/canvas-engine/api';
import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it } from 'vitest';

import { canExportRasterPsd, getStackActions, isStackAllVisible, planStackVisibilityToggle } from './layerStackActions';

const raster = (source: CanvasLayerSourceContract, id = 'r'): CanvasLayerContract =>
  layerContract(id, 'raster', { source });
const imageRef = { height: 10, imageName: 'img', width: 10 };
const entriesOf = (nodes: CanvasNodeContract[]) => compileDocumentNodes({ stacks: stacksFrom(nodes) });

describe('getStackActions', () => {
  it('offers merge-visible + export-psd only to raster, hide-all only to overlay stacks', () => {
    expect(getStackActions('raster')).toEqual(['mergeVisible', 'exportPsd', 'new']);
    for (const stack of ['control', 'inpaint_mask', 'regional_guidance'] as const) {
      expect(getStackActions(stack)).toEqual(['toggleVisibility', 'new']);
    }
  });
});

describe('canExportRasterPsd', () => {
  it('counts pixel-backed and rasterizable sources, not masks or polygons', () => {
    expect(canExportRasterPsd([])).toBe(false);
    expect(canExportRasterPsd([layerContract('m', 'inpaint_mask')])).toBe(false);
    expect(canExportRasterPsd([raster({ bitmap: null, type: 'paint' })])).toBe(true);
    expect(canExportRasterPsd([raster({ image: imageRef, type: 'image' })])).toBe(true);
    expect(
      canExportRasterPsd([
        raster({
          fill: null,
          height: 10,
          kind: 'polygon',
          points: [],
          stroke: null,
          strokeWidth: 0,
          type: 'shape',
          width: 10,
        }),
      ])
    ).toBe(false);
  });
});

describe('planStackVisibilityToggle', () => {
  it('turns the roots off when every leaf is visible', () => {
    const entries = entriesOf([
      raster({ bitmap: null, type: 'paint' }, 'a'),
      groupContract('g', [raster({ bitmap: null, type: 'paint' }, 'b')]),
    ]);
    expect(isStackAllVisible(entries, 'enabled')).toBe(true);
    expect(planStackVisibilityToggle(entries, 'enabled')).toEqual({ ids: ['a', 'g'], nextVisible: false });
  });

  it('turns every node that is off in its own right back on, so a disabled group cannot keep its leaves dark', () => {
    const entries = entriesOf([
      groupContract(
        'g',
        [raster({ bitmap: null, type: 'paint' }, 'b'), layerContract('c', 'raster', { isEnabled: false })],
        { isEnabled: false }
      ),
    ]);
    expect(isStackAllVisible(entries, 'enabled')).toBe(false);
    expect(planStackVisibilityToggle(entries, 'enabled')).toEqual({ ids: ['g', 'c'], nextVisible: true });
  });

  it('drives the display axis of overlay stacks through group hidden flags', () => {
    const hiddenGroup = { ...groupContract('g', [layerContract('c1', 'control')]), isHidden: true };
    const entries = entriesOf([hiddenGroup, layerContract('c2', 'control')]);
    expect(isStackAllVisible(entries, 'hidden')).toBe(false);
    expect(planStackVisibilityToggle(entries, 'hidden')).toEqual({ ids: ['g'], nextVisible: true });
    const shown = entriesOf([groupContract('g', [layerContract('c1', 'control')]), layerContract('c2', 'control')]);
    expect(planStackVisibilityToggle(shown, 'hidden')).toEqual({ ids: ['g', 'c2'], nextVisible: false });
  });
});
