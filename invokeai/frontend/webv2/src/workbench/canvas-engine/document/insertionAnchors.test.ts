import type { CanvasNodeContract, CanvasStackForests } from '@workbench/canvas-engine/contracts';

import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it } from 'vitest';

import { getDocumentIndex } from './documentIndex';
import { removeNodes } from './documentTree';
import {
  captureInsertionAnchor,
  captureRestoreAnchor,
  insertNodesAtAnchor,
  resolveInsertionTarget,
} from './insertionAnchors';

const layer = layerContract;
const group = groupContract;

/**
 * raster: r1, g1[r2, g2[r3], r4], r5  ·  control: c1
 */
const stacks = (): CanvasStackForests =>
  stacksFrom([
    layer('r1'),
    group('g1', [layer('r2'), group('g2', [layer('r3')]), layer('r4')]),
    layer('r5'),
    layer('c1', 'control'),
  ]);

const capture = { editRevision: 7, projectId: 'p' };
const preorder = (forest: CanvasStackForests, stack: 'raster' | 'control' = 'raster'): string[] =>
  getDocumentIndex({ stacks: forest })
    .nodes.filter((entry) => entry.stack === stack)
    .map((entry) => `${'  '.repeat(entry.path.length)}${entry.node.id}`);

describe('captureInsertionAnchor', () => {
  it('anchors above a node under its own parent and remembers the sibling above it', () => {
    expect(captureInsertionAnchor(stacks(), { ...capture, aboveId: 'r4', stack: 'raster' })).toEqual({
      afterId: 'g2',
      beforeId: 'r4',
      capturedEditRevision: 7,
      parentPath: ['g1'],
      projectId: 'p',
      stack: 'raster',
    });
    expect(captureInsertionAnchor(stacks(), { ...capture, aboveId: 'r3', stack: 'raster' })).toMatchObject({
      afterId: null,
      beforeId: 'r3',
      parentPath: ['g1', 'g2'],
    });
  });

  it('anchors inside a group at its top when asked to', () => {
    expect(captureInsertionAnchor(stacks(), { ...capture, insideId: 'g1', stack: 'raster' })).toMatchObject({
      afterId: null,
      beforeId: 'r2',
      parentPath: ['g1'],
    });
    expect(
      captureInsertionAnchor(stacks(), { ...capture, aboveId: 'r5', insideId: 'r1', stack: 'raster' })
    ).toMatchObject({ beforeId: 'r5', parentPath: [] });
  });

  it('falls back to the stack top for an incompatible, absent, or omitted node', () => {
    const top = {
      afterId: null,
      beforeId: 'r1',
      capturedEditRevision: 7,
      parentPath: [],
      projectId: 'p',
      stack: 'raster',
    };

    expect(captureInsertionAnchor(stacks(), { ...capture, aboveId: 'c1', stack: 'raster' })).toEqual(top);
    expect(captureInsertionAnchor(stacks(), { ...capture, aboveId: 'ghost', stack: 'raster' })).toEqual(top);
    expect(captureInsertionAnchor(stacks(), { ...capture, stack: 'raster' })).toEqual(top);
    expect(captureInsertionAnchor(stacks(), { ...capture, stack: 'regional_guidance' })).toMatchObject({
      afterId: null,
      beforeId: null,
      parentPath: [],
    });
  });
});

describe('captureRestoreAnchor', () => {
  it('captures the siblings on both sides and the parent path', () => {
    expect(captureRestoreAnchor(stacks(), 'g2', 'p', 3)).toEqual({
      afterId: 'r2',
      beforeId: 'r4',
      capturedEditRevision: 3,
      parentPath: ['g1'],
      projectId: 'p',
      stack: 'raster',
    });
    expect(captureRestoreAnchor(stacks(), 'c1', 'p', 3)).toMatchObject({
      afterId: null,
      beforeId: null,
      stack: 'control',
    });
    expect(captureRestoreAnchor(stacks(), 'ghost', 'p', 3)).toBeNull();
  });
});

describe('resolveInsertionTarget', () => {
  const anchor = captureInsertionAnchor(stacks(), { ...capture, aboveId: 'r4', stack: 'raster' });

  it('lands before a surviving beforeId under its current parent', () => {
    expect(resolveInsertionTarget(stacks(), anchor)).toEqual({ index: 2, parentId: 'g1' });
  });

  it('follows a moved beforeId to its new parent', () => {
    const moved = stacksFrom([
      layer('r4'),
      layer('r1'),
      group('g1', [layer('r2'), group('g2', [layer('r3')])]),
      layer('r5'),
    ]);
    expect(resolveInsertionTarget(moved, anchor)).toEqual({ index: 0, parentId: null });
  });

  it('lands after a surviving afterId once beforeId is gone', () => {
    expect(resolveInsertionTarget(removeNodes(stacks(), new Set(['r4'])), anchor)).toEqual({
      index: 2,
      parentId: 'g1',
    });
  });

  it('lands at the top of the captured parent, then the nearest surviving ancestor, then the stack top', () => {
    const inner = captureInsertionAnchor(stacks(), { ...capture, aboveId: 'r3', stack: 'raster' });
    expect(resolveInsertionTarget(removeNodes(stacks(), new Set(['r3'])), inner)).toEqual({ index: 0, parentId: 'g2' });
    expect(resolveInsertionTarget(removeNodes(stacks(), new Set(['g2'])), inner)).toEqual({ index: 0, parentId: 'g1' });
    expect(resolveInsertionTarget(removeNodes(stacks(), new Set(['g1'])), inner)).toEqual({ index: 0, parentId: null });
  });

  it('ignores ids that moved to another stack', () => {
    const retyped = stacksFrom([layer('r1'), group('g1', [layer('r2')]), layer('r4', 'control')]);
    expect(resolveInsertionTarget(retyped, anchor)).toEqual({ index: 0, parentId: 'g1' });
  });
});

describe('insertNodesAtAnchor', () => {
  it('splices the nodes at the resolved target and shares every untouched node', () => {
    const before = stacks();
    const anchor = captureInsertionAnchor(before, { ...capture, aboveId: 'r4', stack: 'raster' });
    const after = insertNodesAtAnchor(before, anchor, [layer('n1'), layer('n2')]);

    expect(preorder(after)).toEqual(['r1', 'g1', '  r2', '  g2', '    r3', '  n1', '  n2', '  r4', 'r5']);
    expect(after.control).toBe(before.control);
    expect(after.raster[0]).toBe(before.raster[0]);
    expect((after.raster[1] as { children: CanvasNodeContract[] }).children[1]).toBe(
      (before.raster[1] as { children: CanvasNodeContract[] }).children[1]
    );
  });
});
