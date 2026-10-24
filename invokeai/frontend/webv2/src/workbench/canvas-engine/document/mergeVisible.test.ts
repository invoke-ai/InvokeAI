import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';

import {
  documentFrom,
  groupContract,
  layerContract,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { compileDocumentLeaves } from '@workbench/canvas-engine/document-model/documentModel';
import { describe, expect, it } from 'vitest';

import {
  areSelectedRasterLayersContiguous,
  canMergeInpaintMasks,
  canMergeSelectedRasters,
  canMergeVisibleRasters,
  getMergeVisibleRasterLayers,
} from './mergeVisible';

const raster = (id: string, overrides: Partial<CanvasLayerContract> = {}) => layerContract(id, 'raster', overrides);
const mask = (id: string) => layerContract(id, 'inpaint_mask');
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
  } as Partial<CanvasLayerContract>);

const hasContent = (id: string): boolean => id !== 'empty';
const leavesOf = (nodes: Parameters<typeof documentFrom>[0]) => compileDocumentLeaves(documentFrom(nodes));

describe('getMergeVisibleRasterLayers', () => {
  it('returns every contributing raster with content in stack order, descending into groups', () => {
    const leaves = leavesOf([
      raster('top'),
      mask('mask'),
      groupContract('g', [raster('inner'), raster('hidden', { isEnabled: false })]),
      groupContract('off', [raster('gated')], { isEnabled: false }),
      raster('locked', { isLocked: true }),
      gradientRaster('gradient'),
      raster('empty'),
      raster('bottom'),
    ]);

    expect(getMergeVisibleRasterLayers(leaves, hasContent).map((layer) => layer.id)).toEqual([
      'top',
      'inner',
      'locked',
      'gradient',
      'bottom',
    ]);
    expect(canMergeVisibleRasters(leaves, hasContent)).toBe(true);
  });

  it('requires at least two contributing raster layers with content', () => {
    expect(canMergeVisibleRasters(leavesOf([raster('one')]), hasContent)).toBe(false);
    expect(canMergeVisibleRasters(leavesOf([raster('one'), raster('hidden', { isEnabled: false })]), hasContent)).toBe(
      false
    );
    expect(canMergeVisibleRasters(leavesOf([raster('one'), raster('empty')]), hasContent)).toBe(false);
    expect(canMergeVisibleRasters([], hasContent)).toBe(false);
  });
});

describe('merge-selected eligibility', () => {
  const check = (nodes: Parameters<typeof documentFrom>[0], ids: string[]) => {
    const document = documentFrom(nodes);
    const selected = new Set(ids);
    return {
      contiguous: areSelectedRasterLayersContiguous(document, selected),
      mergeable: canMergeSelectedRasters(document, compileDocumentLeaves(document), selected, hasContent),
    };
  };

  it('accepts adjacent raster siblings at the root or inside one group', () => {
    expect(check([raster('top'), raster('bottom'), mask('mask')], ['top', 'bottom'])).toEqual({
      contiguous: true,
      mergeable: true,
    });
    expect(check([groupContract('g', [raster('a'), raster('b')])], ['a', 'b'])).toEqual({
      contiguous: true,
      mergeable: true,
    });
  });

  it('rejects a selection spanning an unselected sibling, a group, or two parents', () => {
    expect(check([raster('top'), raster('middle'), raster('bottom')], ['top', 'bottom']).contiguous).toBe(false);
    expect(
      check([raster('top'), groupContract('g', [raster('x')]), raster('bottom')], ['top', 'bottom']).contiguous
    ).toBe(false);
    expect(check([raster('top'), groupContract('g', [raster('inner')])], ['top', 'inner']).contiguous).toBe(false);
    expect(check([raster('top'), groupContract('g', [raster('inner')])], ['top', 'g']).contiguous).toBe(false);
  });

  it('rejects empty, locked, gated, disabled, or non-normal raster contributors', () => {
    expect(check([raster('top'), raster('empty')], ['top', 'empty']).mergeable).toBe(false);
    expect(check([raster('top'), raster('bottom', { isLocked: true })], ['top', 'bottom']).mergeable).toBe(false);
    expect(check([raster('top'), raster('bottom', { isEnabled: false })], ['top', 'bottom']).mergeable).toBe(false);
    expect(check([raster('top'), raster('bottom', { blendMode: 'multiply' })], ['top', 'bottom']).mergeable).toBe(
      false
    );
    expect(check([groupContract('g', [raster('a'), raster('b')], { isLocked: true })], ['a', 'b']).mergeable).toBe(
      false
    );
    expect(check([groupContract('g', [raster('a'), raster('b')], { isEnabled: false })], ['a', 'b']).mergeable).toBe(
      false
    );
  });
});

describe('merge-masks eligibility', () => {
  const nodes = [
    mask('top'),
    groupContract('a', [mask('inA')]),
    groupContract('b', [mask('inB')]),
    groupContract('lockedGroup', [mask('underLock')], { isLocked: true }),
    layerContract('disabled', 'inpaint_mask', { isEnabled: false }),
    layerContract('locked', 'inpaint_mask', { isLocked: true }),
    layerContract('region', 'regional_guidance'),
    mask('empty'),
    raster('raster'),
  ];
  const can = (ids: string[]) => canMergeInpaintMasks(leavesOf(nodes), new Set(ids), hasContent);

  it('accepts two or more contributing, unlocked masks with pixels in any groups', () => {
    expect(can(['top', 'inA', 'inB'])).toBe(true);
    expect(can(['inA', 'inB'])).toBe(true);
  });

  it.each([
    ['a single mask', ['top']],
    ['a regional guidance layer', ['top', 'region']],
    ['a raster', ['top', 'raster']],
    ['a group', ['top', 'a']],
    ['a missing id', ['top', 'gone']],
    ['a disabled mask', ['top', 'disabled']],
    ['a locked mask', ['top', 'locked']],
    ['a mask under a locked group', ['top', 'underLock']],
    ['a mask without pixels', ['top', 'empty']],
  ])('refuses a selection holding %s', (_, ids) => {
    expect(can(ids)).toBe(false);
  });
});
