import type { CanvasDocumentContractV3, CanvasNodeContract } from '@workbench/canvas-engine/contracts';
import type { Project } from '@workbench/projectContracts';

import { CANVAS_MAX_NODE_COUNT } from '@workbench/canvas-engine/contracts';
import { getDocumentIndex, getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { isGroupNode } from '@workbench/canvas-engine/document/documentTree';
import { isHideableLayer } from '@workbench/canvas-engine/document/layerEligibility';
import { haveSameStructure } from '@workbench/canvas-engine/document/layerStacks';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { describe, expect, it } from 'vitest';

import type { DocumentCommand, PreparedDocumentEdit, PrepareEditResult } from './documentCommands';

import { createLargeFlatDocument, groupContract, layerContract, stacksFrom } from './documentFixtures.testStub';
import {
  compileDocumentLeaves,
  createDocumentModel,
  getDocumentModelDiagnostics,
  lookupDocumentLayer,
  lookupDocumentLeaf,
  lookupLayerBelow,
  mergeDownEligibility,
} from './documentModel';
import { checkEditPostconditions } from './postconditions';

const layer = layerContract;
const group = groupContract;

/** inpaint: i1 · regional: g1 · control: c1 · raster: r1, r2, r3 */
const flat = (): CanvasNodeContract[] => [
  layer('i1', 'inpaint_mask'),
  layer('r1'),
  layer('c1', 'control'),
  layer('r2'),
  layer('g1', 'regional_guidance'),
  layer('r3'),
];

/** inpaint: i1 · control: c1 · raster: r1, G[r2, H[r3], r4], r5 */
const tree = (): CanvasNodeContract[] => [
  layer('i1', 'inpaint_mask'),
  layer('c1', 'control'),
  layer('r1'),
  group('G', [layer('r2'), group('H', [layer('r3')]), layer('r4')]),
  layer('r5'),
];

const projectWith = (nodes: CanvasNodeContract[], selectedLayerId: string | null): Project => {
  const initial = createInitialWorkbenchState().projects[0]!;
  const project = applyCanvasProjectMutation(initial, {
    document: { ...createEmptyCanvasDocument(), selectedLayerId, stacks: stacksFrom(nodes) },
    type: 'replaceCanvasDocument',
  });
  if (
    project.canvas.document.stacks.raster.length + project.canvas.document.stacks.control.length === 0 &&
    nodes.length > 0
  ) {
    throw new Error('fixture document was not accepted by the reducer');
  }
  return project;
};

const context = (project: Project) => ({ editRevision: 0, projectId: project.id });
const modelOf = (project: Project) => createDocumentModel(project.canvas.document, context(project));

/** Every node id, stacks top first, each in preorder with indentation for depth. */
const outline = (document: CanvasDocumentContractV3): string[] =>
  getDocumentIndex(document).nodes.map((entry) => `${'  '.repeat(entry.path.length)}${entry.node.id}`);

/** The reducer writes an explicit `isHidden: false` on undo where the original node had no key. */
const withExplicitHidden = (node: CanvasNodeContract): CanvasNodeContract =>
  isGroupNode(node)
    ? { ...node, children: node.children.map(withExplicitHidden), isHidden: node.isHidden === true }
    : isHideableLayer(node)
      ? { ...node, isHidden: node.isHidden === true }
      : node;
const normalized = (document: CanvasDocumentContractV3) => ({
  ...document,
  stacks: Object.fromEntries(
    Object.entries(document.stacks).map(([stack, nodes]) => [stack, nodes.map(withExplicitHidden)])
  ),
});

const roundTrip = (project: Project, command: DocumentCommand): { after: Project; edit: PreparedDocumentEdit } => {
  const result = modelOf(project).prepare(command);
  if (result.status !== 'prepared') {
    throw new Error(`expected a prepared edit, got ${JSON.stringify(result)}`);
  }
  const after = applyCanvasProjectMutation(project, result.edit.forward);
  expect(after.canvas.document).not.toBe(project.canvas.document);
  expect(checkEditPostconditions(after.canvas.document, result.edit.postconditions)).toBe(true);
  expect(after.canvas.document.selectedLayerId).toBe(result.edit.selectionAfter);
  const restored = applyCanvasProjectMutation(after, result.edit.inverse).canvas.document;
  const original = project.canvas.document;
  expect(haveSameStructure(restored.stacks, original.stacks)).toBe(true);
  expect(normalized(restored)).toEqual(normalized(original));
  return { after, edit: result.edit };
};

describe('createDocumentModel', () => {
  it('indexes lookup, entries and stack roots once per forest identity', () => {
    const project = projectWith(tree(), 'r2');
    const model = modelOf(project);
    const again = modelOf(project);

    expect(model.getLayer('r3')?.id).toBe('r3');
    expect(model.getLayer('G')).toBeNull();
    expect(model.getNode('G')?.type).toBe('group');
    expect(model.getLayer('ghost')).toBeNull();
    expect(model.getEntry('r3')).toMatchObject({ parentId: 'H', path: ['G', 'H'], siblingIndex: 0 });
    expect(model.getStack('raster').map((entry) => entry.id)).toEqual(['r1', 'G', 'r5']);
    expect(again.getStack('raster')).toBe(model.getStack('raster'));
  });

  it('compiles semantic leaves in document order with ancestor state applied', () => {
    const project = projectWith(
      [
        layer('i1', 'inpaint_mask', { isHidden: true } as never),
        group('hidden', [layer('c1', 'control')], { isHidden: true }),
        group('off', [group('locked', [layer('r1', 'raster', { isLocked: true })], { isLocked: true }), layer('r2')], {
          isEnabled: false,
        }),
        layer('r3'),
      ],
      'r1'
    );
    const leaves = modelOf(project).compileLeaves();

    expect(leaves.map((leaf) => [leaf.id, leaf.stack, leaf.parentIds])).toEqual([
      ['i1', 'inpaint_mask', []],
      ['c1', 'control', ['hidden']],
      ['r1', 'raster', ['off', 'locked']],
      ['r2', 'raster', ['off']],
      ['r3', 'raster', []],
    ]);
    expect(leaves[0]).toMatchObject({
      contributionEnabled: true,
      documentHidden: true,
      effectiveLocked: false,
      locked: false,
    });
    expect(leaves[1]).toMatchObject({ contributionEnabled: true, documentHidden: true });
    expect(leaves[2]).toMatchObject({ contributionEnabled: false, effectiveLocked: true, locked: true });
    expect(leaves[3]).toMatchObject({ contributionEnabled: false, effectiveLocked: false, locked: false });
    expect(leaves[4]).toMatchObject({ contributionEnabled: true, documentHidden: false, effectiveLocked: false });
    expect(leaves[4]!.worldTransform).toEqual({ a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
  });

  it('keeps leaf identity across edits that leave a leaf and its ancestors alone', () => {
    const project = projectWith(tree(), 'r2');
    const leaves = modelOf(project).compileLeaves();
    const before = getDocumentModelDiagnostics();

    const selected = applyCanvasProjectMutation(project, { id: 'r1', type: 'setCanvasSelectedLayer' });
    expect(createDocumentModel(selected.canvas.document, context(project)).compileLeaves()).toBe(leaves);
    expect(getDocumentModelDiagnostics().leafCompilations).toBe(before.leafCompilations);

    const renamed = applyCanvasProjectMutation(selected, {
      id: 'r3',
      patch: { name: 'Renamed' },
      type: 'updateCanvasLayer',
    });
    const next = createDocumentModel(renamed.canvas.document, context(project)).compileLeaves();
    expect(getDocumentModelDiagnostics().leavesCompiled).toBe(before.leavesCompiled + 1);
    expect(next.filter((leaf, index) => leaf === leaves[index]).map((leaf) => leaf.id)).toEqual([
      'i1',
      'c1',
      'r1',
      'r2',
      'r4',
      'r5',
    ]);

    const groupRenamed = applyCanvasProjectMutation(renamed, {
      id: 'G',
      patch: { name: 'Folder' },
      type: 'updateCanvasLayer',
    });
    expect(createDocumentModel(groupRenamed.canvas.document, context(project)).compileLeaves()).toEqual(next);
    expect(getDocumentModelDiagnostics().leavesCompiled).toBe(before.leavesCompiled + 1);

    const gated = applyCanvasProjectMutation(groupRenamed, {
      type: 'setCanvasLayersEnabled',
      updates: [{ id: 'H', isEnabled: false }],
    });
    const afterGate = createDocumentModel(gated.canvas.document, context(project)).compileLeaves();
    expect(getDocumentModelDiagnostics().leavesCompiled).toBe(before.leavesCompiled + 2);
    expect(afterGate.find((leaf) => leaf.id === 'r3')).toMatchObject({ contributionEnabled: false });
    expect(afterGate.filter((leaf) => leaf.id !== 'r3').every((leaf) => next.includes(leaf))).toBe(true);

    const reordered = applyCanvasProjectMutation(gated, {
      orders: [{ orderedIds: ['r5', 'G', 'r1'], parentId: null, stack: 'raster' }],
      type: 'reorderCanvasSiblings',
    });
    expect(
      createDocumentModel(reordered.canvas.document, context(project))
        .compileLeaves()
        .every((leaf) => afterGate.includes(leaf))
    ).toBe(true);
  });

  describe('prepare', () => {
    it('inserts above a node under its parent and restores the previous selection on undo', () => {
      const project = projectWith(tree(), 'r3');
      const { after, edit } = roundTrip(project, { aboveId: 'r3', nodes: [layer('n1')], type: 'insert' });

      expect(outline(after.canvas.document)).toEqual([
        'i1',
        'c1',
        'r1',
        'G',
        '  r2',
        '  H',
        '    n1',
        '    r3',
        '  r4',
        'r5',
      ]);
      expect(edit).toMatchObject({
        createdIds: ['n1'],
        history: 'record',
        postconditions: [
          { kind: 'sibling-order', orderedIds: ['n1', 'r3'], parentId: 'H', stack: 'raster' },
          { id: 'n1', kind: 'selection' },
        ],
        selectionAfter: 'n1',
        touchedIds: ['n1'],
        touchedStacks: ['raster'],
      });
    });

    it('inserts inside a group, at a foreign stack top, and honours an explicit selection', () => {
      const project = projectWith(tree(), 'r3');
      const inside = roundTrip(project, { aboveId: null, insideId: 'G', nodes: [layer('n1')], type: 'insert' });
      expect(outline(inside.after.canvas.document).slice(3, 6)).toEqual(['G', '  n1', '  r2']);

      const foreign = roundTrip(project, {
        aboveId: 'r3',
        nodes: [layer('m2', 'inpaint_mask')],
        selectId: null,
        type: 'insert',
      });
      expect(outline(foreign.after.canvas.document).slice(0, 2)).toEqual(['m2', 'i1']);
      expect(foreign.after.canvas.document.selectedLayerId).toBeNull();
    });

    it('inserts a whole subtree and refuses one that mixes stacks or nests too deep', () => {
      const project = projectWith(tree(), 'r1');
      const subtree = group('S', [layer('s1'), group('S2', [layer('s2')])]);
      const { after, edit } = roundTrip(project, { aboveId: 'r5', nodes: [subtree], type: 'insert' });
      expect(outline(after.canvas.document).slice(-5)).toEqual(['S', '  s1', '  S2', '    s2', 'r5']);
      expect(edit.createdIds).toEqual(['S', 's1', 'S2', 's2']);

      expect(
        modelOf(project).prepare({
          aboveId: null,
          nodes: [group('bad', [layer('x'), layer('y', 'control')])],
          type: 'insert',
        })
      ).toEqual({
        reason: 'foreign-stack',
        status: 'invalid-target',
        targetId: 'bad',
      });
      const deep = (depth: number): CanvasNodeContract =>
        depth === 0 ? layer(`d${depth}`) : group(`d${depth}`, [deep(depth - 1)]);
      expect(modelOf(project).prepare({ aboveId: 'r3', nodes: [deep(9)], type: 'insert' })).toEqual({
        reason: 'depth-exceeded',
        status: 'invalid-target',
        targetId: 'd9',
      });
      expect(modelOf(project).prepare({ aboveId: null, nodes: [deep(9)], type: 'insert' }).status).toBe('prepared');
    });

    it('removes nodes, repairs the selection to a sibling, and restores runs between their neighbours', () => {
      const project = projectWith(tree(), 'r2');
      const { after, edit } = roundTrip(project, { ids: ['r2', 'i1', 'r4'], type: 'remove' });

      expect(outline(after.canvas.document)).toEqual(['c1', 'r1', 'G', '  H', '    r3', 'r5']);
      expect(edit).toMatchObject({
        selectionAfter: 'H',
        touchedIds: ['i1', 'r2', 'r4'],
        touchedStacks: ['inpaint_mask', 'raster'],
      });
    });

    it('removes a group with its descendants, once, even when descendants are listed too', () => {
      const project = projectWith(tree(), 'r3');
      const { after, edit } = roundTrip(project, { ids: ['G', 'r3'], type: 'remove' });

      expect(outline(after.canvas.document)).toEqual(['i1', 'c1', 'r1', 'r5']);
      expect(edit.forward).toEqual({ ids: ['G'], type: 'removeCanvasLayers' });
      expect(edit.touchedIds).toEqual(['G', 'r2', 'H', 'r3', 'r4']);
      expect(edit.selectionAfter).toBe('r5');
    });

    it('refuses to remove a node with a locked ancestor or descendant', () => {
      const project = projectWith(
        [group('G', [layer('r1'), group('H', [layer('r2', 'raster', { isLocked: true })])], { isLocked: true })],
        'r1'
      );
      expect(modelOf(project).prepare({ ids: ['r1'], type: 'remove' })).toEqual({ ids: ['G'], status: 'locked' });
      expect(modelOf(project).prepare({ ids: ['G'], type: 'remove' })).toEqual({ ids: ['G', 'r2'], status: 'locked' });
    });

    it('duplicates leaves and subtrees above their sources with fresh ids', () => {
      let next = 0;
      const project = projectWith(tree(), 'r1');
      const { after, edit } = roundTrip(project, {
        createId: () => `d${next++}`,
        ids: ['G', 'r3', 'r5'],
        type: 'duplicate',
      });

      expect(outline(after.canvas.document)).toEqual([
        'i1',
        'c1',
        'r1',
        'd0',
        '  d1',
        '  d2',
        '    d3',
        '  d4',
        'G',
        '  r2',
        '  H',
        '    r3',
        '  r4',
        'd5',
        'r5',
      ]);
      expect(edit).toMatchObject({ createdIds: ['d0', 'd1', 'd2', 'd3', 'd4', 'd5'], selectionAfter: 'd5' });
      expect(after.canvas.document.stacks.raster[1]).toMatchObject({ name: 'G copy', type: 'group' });
    });

    it('moves within sibling lists, keeping a selected group and its descendants together', () => {
      const project = projectWith(tree(), 'r3');
      const moved = roundTrip(project, { ids: ['G', 'r3'], kind: 'front', type: 'move' });
      expect(outline(moved.after.canvas.document)).toEqual([
        'i1',
        'c1',
        'G',
        '  r2',
        '  H',
        '    r3',
        '  r4',
        'r1',
        'r5',
      ]);
      expect(moved.edit.touchedIds).toEqual(['G', 'r1']);

      const reordered = roundTrip(project, {
        orders: [
          { orderedIds: ['r4', 'r2', 'H'], parentId: 'G', stack: 'raster' },
          { orderedIds: ['c1'], parentId: null, stack: 'control' },
        ],
        type: 'reorder',
      });
      expect(outline(reordered.after.canvas.document)).toEqual([
        'i1',
        'c1',
        'r1',
        'G',
        '  r4',
        '  r2',
        '  H',
        '    r3',
        'r5',
      ]);
      expect(reordered.edit).toMatchObject({ touchedIds: ['r4', 'r2', 'H'], touchedStacks: ['control', 'raster'] });
    });

    it('reparents nodes from several parents into a group before a sibling, and back exactly on undo', () => {
      const project = projectWith(tree(), 'r1');
      const { after, edit } = roundTrip(project, {
        beforeId: 'r3',
        ids: ['r1', 'r4', 'r5'],
        parentId: 'H',
        type: 'reparent',
      });

      expect(outline(after.canvas.document)).toEqual([
        'i1',
        'c1',
        'G',
        '  r2',
        '  H',
        '    r1',
        '    r4',
        '    r5',
        '    r3',
      ]);
      expect(edit.forward).toMatchObject({
        move: [{ anchor: { beforeId: 'r3', parentPath: ['G', 'H'] }, ids: ['r1', 'r4', 'r5'] }],
      });
      expect(edit.inverse).toMatchObject({
        move: [
          { anchor: { afterId: null, beforeId: 'G', parentPath: [] }, ids: ['r1'] },
          { anchor: { afterId: 'G', beforeId: null, parentPath: [] }, ids: ['r5'] },
          { anchor: { afterId: 'H', beforeId: null, parentPath: ['G'] }, ids: ['r4'] },
        ],
      });
      expect(edit.selectionAfter).toBe('r1');
    });

    it('reparents to the bottom of the root and reports an unchanged placement', () => {
      const project = projectWith(tree(), 'r1');
      const { after } = roundTrip(project, { beforeId: null, ids: ['r3'], parentId: null, type: 'reparent' });
      expect(outline(after.canvas.document)).toEqual(['i1', 'c1', 'r1', 'G', '  r2', '  H', '  r4', 'r5', 'r3']);
      expect(modelOf(project).prepare({ beforeId: 'r5', ids: ['G'], parentId: null, type: 'reparent' })).toEqual({
        status: 'unchanged',
      });
      expect(modelOf(project).prepare({ beforeId: null, ids: ['r5'], parentId: null, type: 'reparent' })).toEqual({
        status: 'unchanged',
      });
    });

    it.each<[string, DocumentCommand, PrepareEditResult]>([
      [
        'a cycle',
        { beforeId: null, ids: ['G'], parentId: 'H', type: 'reparent' },
        { reason: 'cycle', status: 'invalid-target', targetId: 'H' },
      ],
      [
        'a leaf parent',
        { beforeId: null, ids: ['r5'], parentId: 'r1', type: 'reparent' },
        { reason: 'not-a-group', status: 'invalid-target', targetId: 'r1' },
      ],
      [
        'a foreign parent',
        { beforeId: null, ids: ['c1'], parentId: 'G', type: 'reparent' },
        { reason: 'foreign-stack', status: 'invalid-target', targetId: 'G' },
      ],
      [
        'mixed stacks',
        { beforeId: null, ids: ['c1', 'r5'], parentId: null, type: 'reparent' },
        { reason: 'foreign-stack', status: 'invalid-target', targetId: 'r5' },
      ],
      [
        'a beforeId outside the target',
        { beforeId: 'r2', ids: ['r5'], parentId: null, type: 'reparent' },
        { reason: 'not-siblings', status: 'invalid-target', targetId: 'r2' },
      ],
      [
        'a moving beforeId',
        { beforeId: 'r5', ids: ['r5', 'r1'], parentId: null, type: 'reparent' },
        { reason: 'not-siblings', status: 'invalid-target', targetId: 'r5' },
      ],
      [
        'a missing parent',
        { beforeId: null, ids: ['r5'], parentId: 'ghost', type: 'reparent' },
        { ids: ['ghost'], status: 'missing' },
      ],
    ])('refuses reparenting with %s', (_label, command, refusal) => {
      expect(modelOf(projectWith(tree(), 'r1')).prepare(command)).toEqual(refusal);
    });

    it('refuses to reparent into a locked group, out of one, or past the depth limit', () => {
      const locked = projectWith([group('L', [layer('r1')], { isLocked: true }), layer('r2')], 'r2');
      expect(modelOf(locked).prepare({ beforeId: null, ids: ['r2'], parentId: 'L', type: 'reparent' })).toEqual({
        ids: ['L'],
        status: 'locked',
      });
      expect(modelOf(locked).prepare({ beforeId: null, ids: ['r1'], parentId: null, type: 'reparent' })).toEqual({
        ids: ['L'],
        status: 'locked',
      });

      const nest = (depth: number): CanvasNodeContract =>
        depth === 0 ? layer('leaf') : group(`n${depth}`, [nest(depth - 1)]);
      const deep = projectWith([nest(10), group('S', [layer('s')])], 'leaf');
      expect(modelOf(deep).prepare({ beforeId: null, ids: ['S'], parentId: 'n1', type: 'reparent' })).toEqual({
        reason: 'depth-exceeded',
        status: 'invalid-target',
        targetId: 'n1',
      });
      expect(modelOf(deep).prepare({ beforeId: null, ids: ['s'], parentId: 'n1', type: 'reparent' }).status).toBe(
        'prepared'
      );
    });

    it('groups selected siblings at the topmost one, in order, and dissolves the group exactly on undo', () => {
      const project = projectWith(tree(), 'r5');
      const { after, edit } = roundTrip(project, { groupId: 'N', ids: ['r5', 'r1'], name: 'Group 1', type: 'group' });

      expect(outline(after.canvas.document)).toEqual([
        'i1',
        'c1',
        'N',
        '  r1',
        '  r5',
        'G',
        '  r2',
        '  H',
        '    r3',
        '  r4',
      ]);
      expect(after.canvas.document.stacks.raster[0]).toMatchObject({
        isEnabled: true,
        isLocked: false,
        name: 'Group 1',
        type: 'group',
      });
      expect(edit).toMatchObject({
        createdIds: ['N'],
        selectionAfter: 'N',
        touchedIds: ['N', 'r1', 'r5'],
        touchedStacks: ['raster'],
      });
    });

    it('groups nested siblings and refuses non-siblings, a locked parent, or a taken id', () => {
      const project = projectWith(tree(), 'r2');
      const { after } = roundTrip(project, { groupId: 'N', ids: ['r4', 'H'], name: 'Inner', type: 'group' });
      expect(outline(after.canvas.document)).toEqual([
        'i1',
        'c1',
        'r1',
        'G',
        '  r2',
        '  N',
        '    H',
        '      r3',
        '    r4',
        'r5',
      ]);

      expect(modelOf(project).prepare({ groupId: 'N', ids: ['r1', 'r2'], name: 'x', type: 'group' })).toEqual({
        reason: 'not-siblings',
        status: 'invalid-target',
        targetId: 'r2',
      });
      expect(modelOf(project).prepare({ groupId: 'G', ids: ['r1'], name: 'x', type: 'group' })).toEqual({
        reason: 'id-exists',
        status: 'invalid-target',
        targetId: 'G',
      });
      const locked = projectWith([group('L', [layer('a'), layer('b')], { isLocked: true })], 'a');
      expect(modelOf(locked).prepare({ groupId: 'N', ids: ['a', 'b'], name: 'x', type: 'group' })).toEqual({
        ids: ['L'],
        status: 'locked',
      });
    });

    it('ungroups in place, selecting the first child when the group was selected, and regroups on undo', () => {
      const project = projectWith(tree(), 'G');
      const { after, edit } = roundTrip(project, { ids: ['G'], type: 'ungroup' });

      expect(outline(after.canvas.document)).toEqual(['i1', 'c1', 'r1', 'r2', 'H', '  r3', 'r4', 'r5']);
      expect(edit).toMatchObject({ selectionAfter: 'r2', touchedIds: ['G', 'r2', 'H', 'r4'] });

      // Naming a group and one nested inside it dissolves both; nothing half-applies.
      const nested = roundTrip(projectWith(tree(), 'H'), { ids: ['H', 'G'], type: 'ungroup' });
      expect(outline(nested.after.canvas.document)).toEqual(['i1', 'c1', 'r1', 'r2', 'r3', 'r4', 'r5']);
      expect(nested.edit.selectionAfter).toBe('r3');
      expect(modelOf(project).prepare({ ids: ['r1'], type: 'ungroup' })).toEqual({
        actual: 'raster',
        expected: ['group'],
        status: 'wrong-type',
      });
    });

    it('translates leaves and groups by one delta and refuses any locked node in the set', () => {
      const project = projectWith(tree(), 'r1');
      const { after, edit } = roundTrip(project, { dx: 5, dy: -2, ids: ['G', 'r1'], type: 'translate' });
      const moved = getDocumentLeaves(after.canvas.document).filter((leaf) =>
        ['r1', 'r2', 'r3', 'r4'].includes(leaf.id)
      );
      expect(moved.map((leaf) => [leaf.transform.x, leaf.transform.y])).toEqual([
        [5, -2],
        [5, -2],
        [5, -2],
        [5, -2],
      ]);
      expect(lookupDocumentLayer(after.canvas.document, 'r5')?.transform).toMatchObject({ x: 0, y: 0 });
      expect(edit.touchedIds).toEqual(['r1', 'r2', 'r3', 'r4']);

      expect(modelOf(project).prepare({ dx: 0, dy: 0, ids: ['G'], type: 'translate' })).toEqual({
        status: 'unchanged',
      });
      const locked = projectWith([group('G', [layer('a'), layer('b', 'raster', { isLocked: true })])], 'a');
      expect(modelOf(locked).prepare({ dx: 1, dy: 0, ids: ['G'], type: 'translate' })).toEqual({
        ids: ['b'],
        status: 'locked',
      });
      expect(modelOf(locked).prepare({ dx: 1, dy: 0, ids: ['a', 'b'], type: 'translate' })).toEqual({
        ids: ['b'],
        status: 'locked',
      });
    });

    it('patches base fields and transforms with an exact inverse', () => {
      const project = projectWith(flat(), 'r1');
      const { after, edit } = roundTrip(project, {
        id: 'r1',
        patch: { name: 'Renamed', opacity: 0.5, transform: { x: 12 } },
        type: 'patch',
      });

      expect(lookupDocumentLayer(after.canvas.document, 'r1')).toMatchObject({
        name: 'Renamed',
        opacity: 0.5,
        transform: { x: 12, y: 0 },
      });
      expect(edit.inverse).toEqual({
        id: 'r1',
        patch: { name: 'r1', opacity: 1, transform: { x: 0 } },
        type: 'updateCanvasLayer',
      });
      expect(edit.postconditions).toEqual([
        { id: 'r1', kind: 'patched', patch: { name: 'Renamed', opacity: 0.5, transform: { x: 12 } } },
      ]);
    });

    it('patches a group name, enabled and locked state, and refuses leaf-only fields on a group', () => {
      const project = projectWith(tree(), 'G');
      const { after, edit } = roundTrip(project, {
        id: 'G',
        patch: { isEnabled: false, name: 'Folder' },
        type: 'patch',
      });
      expect(after.canvas.document.stacks.raster[1]).toMatchObject({ isEnabled: false, name: 'Folder' });
      expect(edit.inverse).toEqual({ id: 'G', patch: { isEnabled: true, name: 'G' }, type: 'updateCanvasLayer' });
      expect(modelOf(project).prepare({ id: 'G', patch: { transform: { x: 1 } }, type: 'patch' })).toMatchObject({
        status: 'wrong-type',
      });
    });

    it('patches opacity and blend on a raster-stack group, with an inverse restoring absence', () => {
      const project = projectWith(tree(), 'G');
      const { after, edit } = roundTrip(project, {
        id: 'G',
        patch: { blendMode: 'multiply', opacity: 0.5 },
        type: 'patch',
      });
      expect(after.canvas.document.stacks.raster[1]).toMatchObject({ blendMode: 'multiply', opacity: 0.5 });
      expect(edit.inverse).toEqual({
        id: 'G',
        patch: { blendMode: undefined, opacity: undefined },
        type: 'updateCanvasLayer',
      });
    });

    it('patches a color label on leaves and groups in any stack, locked included, and clears to absence', () => {
      const project = projectWith(
        [layer('r1', 'raster', { isLocked: true }), group('OG', [layer('c1', 'control')])],
        null
      );
      // Organizational, so the lock does not gate it.
      const { after, edit } = roundTrip(project, { id: 'r1', patch: { colorLabel: 'red' }, type: 'patch' });
      expect(after.canvas.document.stacks.raster[0]).toMatchObject({ colorLabel: 'red' });
      expect(edit.inverse).toEqual({ id: 'r1', patch: { colorLabel: undefined }, type: 'updateCanvasLayer' });

      const overlayGroup = roundTrip(project, { id: 'OG', patch: { colorLabel: 'blue' }, type: 'patch' });
      expect(overlayGroup.after.canvas.document.stacks.control[0]).toMatchObject({ colorLabel: 'blue' });

      const cleared = applyCanvasProjectMutation(after, {
        id: 'r1',
        patch: { colorLabel: undefined },
        type: 'updateCanvasLayer',
      });
      expect(
        cleared.canvas.document.stacks.raster[0] && 'colorLabel' in cleared.canvas.document.stacks.raster[0]
          ? cleared.canvas.document.stacks.raster[0].colorLabel
          : undefined
      ).toBeUndefined();
    });

    it('refuses opacity and blend on an overlay-stack group and on a locked group', () => {
      const overlay = projectWith([group('OG', [layer('c1', 'control')])], null);
      expect(modelOf(overlay).prepare({ id: 'OG', patch: { opacity: 0.5 }, type: 'patch' })).toEqual({
        operation: 'blend an overlay-stack group',
        status: 'unsupported',
      });
      expect(modelOf(overlay).prepare({ id: 'OG', patch: { blendMode: 'multiply' }, type: 'patch' })).toEqual({
        operation: 'blend an overlay-stack group',
        status: 'unsupported',
      });
      // Renaming a locked group stays allowed; its appearance does not.
      const locked = projectWith([group('LG', [layer('r1')], { isLocked: true })], null);
      expect(modelOf(locked).prepare({ id: 'LG', patch: { opacity: 0.5 }, type: 'patch' })).toEqual({
        ids: ['LG'],
        status: 'locked',
      });
      expect(modelOf(locked).prepare({ id: 'LG', patch: { name: 'renamed' }, type: 'patch' })).toMatchObject({
        status: 'prepared',
      });
    });

    it('takes a pre-gesture baseline so a previewed patch still records a real inverse', () => {
      const previewed = applyCanvasProjectMutation(projectWith(flat(), 'r1'), {
        id: 'r1',
        patch: { opacity: 0.4 },
        type: 'updateCanvasLayer',
      });
      const model = modelOf(previewed);
      expect(model.prepare({ id: 'r1', patch: { opacity: 0.4 }, type: 'patch' })).toEqual({ status: 'unchanged' });
      expect(model.prepare({ before: { opacity: 1 }, id: 'r1', patch: { opacity: 0.4 }, type: 'patch' })).toMatchObject(
        {
          edit: { inverse: { id: 'r1', patch: { opacity: 1 }, type: 'updateCanvasLayer' } },
          status: 'prepared',
        }
      );
      expect(model.prepare({ before: { opacity: 0.4 }, id: 'r1', patch: { opacity: 0.4 }, type: 'patch' })).toEqual({
        status: 'unchanged',
      });
      expect(
        model.prepare({ before: { name: 'r1', opacity: 1 }, id: 'r1', patch: { opacity: 0.4 }, type: 'patch' })
      ).toEqual({
        operation: 'patch baseline names other fields',
        status: 'unsupported',
      });
      expect(
        model.prepare({
          before: { transform: { x: 0 } },
          id: 'r1',
          patch: { transform: { x: 1, y: 2 } },
          type: 'patch',
        })
      ).toEqual({ operation: 'patch baseline names other fields', status: 'unsupported' });
      expect(
        model.prepare({
          before: { layerType: 'inpaint_mask', mask: { fill: { color: '#000000', style: 'solid' } } },
          config: { layerType: 'inpaint_mask', mask: { offset: { x: 4, y: 4 } } },
          id: 'i1',
          type: 'patch-config',
        })
      ).toEqual({ operation: 'config baseline names other fields', status: 'unsupported' });
      expect(
        model.prepare({
          before: { isTransparencyLocked: false, layerType: 'control' } as never,
          config: { isTransparencyLocked: true, layerType: 'raster' },
          id: 'r1',
          type: 'patch-config',
        })
      ).toEqual({ operation: 'config baseline names another layer type', status: 'unsupported' });

      const fillBefore = { color: '#e07575', style: 'diagonal' as const };
      const fillNext = { color: '#00ff00', style: 'diagonal' as const };
      const previewedFill = applyCanvasProjectMutation(projectWith(flat(), 'i1'), {
        config: { layerType: 'inpaint_mask', mask: { fill: fillNext } },
        id: 'i1',
        type: 'updateCanvasLayerConfig',
      });
      expect(
        modelOf(previewedFill).prepare({
          before: { layerType: 'inpaint_mask', mask: { fill: fillBefore } },
          config: { layerType: 'inpaint_mask', mask: { fill: { ...fillNext } } },
          id: 'i1',
          type: 'patch-config',
        })
      ).toMatchObject({
        edit: { inverse: { config: { layerType: 'inpaint_mask', mask: { fill: fillBefore } }, id: 'i1' } },
        status: 'prepared',
      });
      expect(
        modelOf(previewedFill).prepare({
          config: { layerType: 'inpaint_mask', mask: { fill: { ...fillNext } } },
          id: 'i1',
          type: 'patch-config',
        })
      ).toEqual({ status: 'unchanged' });
    });

    it('round-trips a group adjustment stack and refuses overlay, locked, and wrong-type targets', () => {
      const stack = [
        { brightness: 0.2, contrast: 0, id: 'ga1', isEnabled: true, type: 'brightness-contrast' as const },
      ];
      const project = projectWith(
        [
          group('G', [layer('r1')]),
          group('GL', [layer('r2')], { isLocked: true }),
          group('OG', [layer('c2', 'control')]),
        ],
        'r1'
      );

      const done = roundTrip(project, {
        config: { adjustments: stack, layerType: 'group' },
        id: 'G',
        type: 'patch-config',
      });
      const adjusted = getDocumentIndex(done.after.canvas.document).byId.get('G')!.node;
      expect(isGroupNode(adjusted) ? adjusted.adjustments : null).toEqual(stack);

      const model = modelOf(project);
      expect(
        model.prepare({ config: { adjustments: stack, layerType: 'group' }, id: 'OG', type: 'patch-config' })
      ).toEqual({ operation: 'adjust an overlay-stack group', status: 'unsupported' });
      expect(
        model.prepare({ config: { adjustments: stack, layerType: 'group' }, id: 'GL', type: 'patch-config' })
      ).toEqual({ ids: ['GL'], status: 'locked' });
      expect(
        model.prepare({ config: { adjustments: stack, layerType: 'group' }, id: 'r1', type: 'patch-config' })
      ).toEqual({ actual: 'raster', expected: ['group'], status: 'wrong-type' });
      expect(
        model.prepare({ config: { adjustments: stack, layerType: 'raster' }, id: 'G', type: 'patch-config' })
      ).toEqual({
        actual: 'group',
        expected: ['raster', 'control', 'regional_guidance', 'inpaint_mask'],
        status: 'wrong-type',
      });

      // Even unvalidated preview/replay dispatches must reject adjustment stacks on overlay groups.
      const rawDispatch = applyCanvasProjectMutation(project, {
        config: { adjustments: stack, layerType: 'group' },
        id: 'OG',
        type: 'updateCanvasLayerConfig',
      });
      expect(rawDispatch.canvas.document).toBe(project.canvas.document);
    });

    it('round-trips a layer regenerate region through add, toggle and null-remove', () => {
      const project = projectWith(flat(), 'r1');
      const inpaint = {
        isEnabled: true,
        fill: { color: '#e07575', style: 'diagonal' as const },
      };
      const added = roundTrip(project, {
        before: { inpaint: null, layerType: 'raster' },
        config: { inpaint, layerType: 'raster' },
        id: 'r1',
        type: 'patch-config',
      });
      const toggled = roundTrip(added.after, {
        before: { inpaint, layerType: 'raster' },
        config: { inpaint: { ...inpaint, isEnabled: false }, layerType: 'raster' },
        id: 'r1',
        type: 'patch-config',
      });
      const removed = roundTrip(toggled.after, {
        before: { inpaint: { ...inpaint, isEnabled: false }, layerType: 'raster' },
        config: { inpaint: null, layerType: 'raster' },
        id: 'r1',
        type: 'patch-config',
      });
      const layer = getDocumentLeaves(removed.after.canvas.document).find((leaf) => leaf.id === 'r1')!;
      expect(Object.hasOwn(layer, 'inpaint')).toBe(false);
    });

    it('round-trips a mask modifier through add, toggle and null-remove with passing postconditions', () => {
      const project = projectWith(flat(), 'i1');
      const noise = { isEnabled: true, level: 0.25 };
      const added = roundTrip(project, {
        before: { layerType: 'inpaint_mask', noise: null },
        config: { layerType: 'inpaint_mask', noise },
        id: 'i1',
        type: 'patch-config',
      });
      const toggled = roundTrip(added.after, {
        before: { layerType: 'inpaint_mask', noise },
        config: { layerType: 'inpaint_mask', noise: { ...noise, isEnabled: false } },
        id: 'i1',
        type: 'patch-config',
      });
      const removed = roundTrip(toggled.after, {
        before: { layerType: 'inpaint_mask', noise: { ...noise, isEnabled: false } },
        config: { layerType: 'inpaint_mask', noise: null },
        id: 'i1',
        type: 'patch-config',
      });
      const mask = getDocumentLeaves(removed.after.canvas.document).find((leaf) => leaf.id === 'i1')!;
      expect(Object.hasOwn(mask, 'noise')).toBe(false);
    });

    it('round-trips setting and clearing a regional prompt, which the layer stores as null', () => {
      const project = projectWith([layer('g1', 'regional_guidance'), layer('r1')], 'g1');
      const set = roundTrip(project, {
        before: { layerType: 'regional_guidance', negativePrompt: null },
        config: { layerType: 'regional_guidance', negativePrompt: 'blurry' },
        id: 'g1',
        type: 'patch-config',
      });
      const cleared = roundTrip(set.after, {
        before: { layerType: 'regional_guidance', negativePrompt: 'blurry' },
        config: { layerType: 'regional_guidance', negativePrompt: null },
        id: 'g1',
        type: 'patch-config',
      });
      const region = getDocumentLeaves(cleared.after.canvas.document).find((leaf) => leaf.id === 'g1')!;
      expect(region).toMatchObject({ negativePrompt: null });
    });

    it('round-trips an atomic cross-layer config batch and refuses malformed batches', () => {
      const ref = {
        config: {
          beginEndStepPct: [0, 1] as [number, number],
          clipVisionModel: 'ViT-H' as const,
          image: null,
          method: 'full' as const,
          model: null,
          type: 'ip_adapter' as const,
          weight: 1,
        },
        id: 'ref1',
        isEnabled: true,
      };
      const nodes = [
        layer('ga', 'regional_guidance', { referenceImages: [ref] }),
        layer('gb', 'regional_guidance'),
        layer('r1'),
      ];
      const project = projectWith(nodes, 'ga');
      const move = {
        patches: [
          {
            before: { layerType: 'regional_guidance' as const, referenceImages: [ref] },
            config: { layerType: 'regional_guidance' as const, referenceImages: [] },
            id: 'ga',
          },
          {
            before: { layerType: 'regional_guidance' as const, referenceImages: [] },
            config: { layerType: 'regional_guidance' as const, referenceImages: [ref] },
            id: 'gb',
          },
        ],
        type: 'patch-config-batch' as const,
      };
      const { after } = roundTrip(project, move);
      const leaves = getDocumentLeaves(after.canvas.document);
      const refsOf = (id: string) => {
        const found = leaves.find((leaf) => leaf.id === id);
        return found?.type === 'regional_guidance' ? found.referenceImages.map((entry) => entry.id) : null;
      };
      expect(refsOf('ga')).toEqual([]);
      expect(refsOf('gb')).toEqual(['ref1']);

      expect(modelOf(project).prepare({ patches: [], type: 'patch-config-batch' })).toEqual({
        operation: 'patch nothing',
        status: 'unsupported',
      });
      expect(
        modelOf(project).prepare({ patches: [move.patches[0]!, move.patches[0]!], type: 'patch-config-batch' })
      ).toEqual({ operation: 'batch patches one layer twice', status: 'unsupported' });
      const locked = projectWith(
        [layer('ga', 'regional_guidance', { isLocked: true, referenceImages: [ref] }), nodes[1]!, layer('r1')],
        'ga'
      );
      expect(modelOf(locked).prepare(move)).toMatchObject({ status: 'locked' });
      expect(
        modelOf(project).prepare({
          patches: [
            {
              before: { layerType: 'regional_guidance', referenceImages: [ref] },
              config: { layerType: 'regional_guidance', referenceImages: [ref] },
              id: 'ga',
            },
          ],
          type: 'patch-config-batch',
        })
      ).toEqual({ status: 'unchanged' });
    });

    it('round-trips config, source and flag commands through the reducer, groups included', () => {
      const project = projectWith(tree(), 'r1');
      const control = roundTrip(project, {
        config: { adapter: { weight: 0.5 }, layerType: 'control', withTransparencyEffect: true },
        id: 'c1',
        type: 'patch-config',
      });
      expect(lookupDocumentLayer(control.after.canvas.document, 'c1')).toMatchObject({
        adapter: { weight: 0.5 },
        withTransparencyEffect: true,
      });
      expect(control.edit.inverse).toEqual({
        config: { adapter: { weight: 1 }, layerType: 'control', withTransparencyEffect: false },
        id: 'c1',
        type: 'updateCanvasLayerConfig',
      });

      const source = { bitmap: null, offset: { x: 3, y: 4 }, type: 'paint' as const };
      const patched = roundTrip(project, { id: 'r3', source, type: 'patch-source' });
      expect((lookupDocumentLayer(patched.after.canvas.document, 'r3') as { source: unknown }).source).toBe(source);

      const enabled = roundTrip(project, {
        type: 'set-enabled',
        updates: [
          { id: 'G', isEnabled: false },
          { id: 'c1', isEnabled: true },
        ],
      });
      expect(enabled.edit).toMatchObject({ touchedIds: ['G'], touchedStacks: ['raster'] });
      expect(
        compileDocumentLeaves(enabled.after.canvas.document).find((leaf) => leaf.id === 'r3')?.contributionEnabled
      ).toBe(false);

      const hidden = roundTrip(projectWith([group('O', [layer('c2', 'control')])], 'c2'), {
        type: 'set-hidden',
        updates: [{ id: 'O', isHidden: true }],
      });
      expect(hidden.after.canvas.document.stacks.control[0]).toMatchObject({ isHidden: true });

      const locked = roundTrip(project, { type: 'set-locked', updates: [{ id: 'H', isLocked: true }] });
      expect(compileDocumentLeaves(locked.after.canvas.document).find((leaf) => leaf.id === 'r3')).toMatchObject({
        effectiveLocked: true,
        locked: false,
      });
    });

    it.each<[string, DocumentCommand, PrepareEditResult]>([
      [
        'hiding a raster layer',
        { type: 'set-hidden', updates: [{ id: 'r1', isHidden: true }] },
        { actual: 'raster', expected: ['control', 'inpaint_mask', 'regional_guidance'], status: 'wrong-type' },
      ],
      [
        'hiding a raster group',
        { type: 'set-hidden', updates: [{ id: 'G', isHidden: true }] },
        { actual: 'group', expected: ['control', 'inpaint_mask', 'regional_guidance'], status: 'wrong-type' },
      ],
      [
        'a config patch for another layer type',
        { config: { layerType: 'control', withTransparencyEffect: true }, id: 'r1', type: 'patch-config' },
        { actual: 'raster', expected: ['control'], status: 'wrong-type' },
      ],
      [
        'a config patch on a group',
        { config: { layerType: 'raster', isTransparencyLocked: true }, id: 'G', type: 'patch-config' },
        { actual: 'group', expected: ['raster', 'control', 'regional_guidance', 'inpaint_mask'], status: 'wrong-type' },
      ],
      [
        'a source patch on a mask',
        { id: 'i1', source: { bitmap: null, type: 'paint' }, type: 'patch-source' },
        { actual: 'inpaint_mask', expected: ['raster', 'control'], status: 'wrong-type' },
      ],
      [
        'an empty flag update',
        { type: 'set-locked', updates: [] },
        { operation: 'set-locked nothing', status: 'unsupported' },
      ],
      [
        'flags that already hold',
        { type: 'set-enabled', updates: [{ id: 'r1', isEnabled: true }] },
        { status: 'unchanged' },
      ],
    ])('answers %s', (_label, command, result) => {
      expect(modelOf(projectWith(tree(), 'r1')).prepare(command)).toEqual(result);
    });

    it('selects any node without recording history and reports the current selection as unchanged', () => {
      const project = projectWith(tree(), 'r1');
      const { edit } = roundTrip(project, { id: 'G', type: 'select' });
      expect(edit).toMatchObject({ history: 'none', selectionAfter: 'G', selectionBefore: 'r1', touchedIds: [] });
      expect(modelOf(project).prepare({ id: 'r1', type: 'select' })).toEqual({ status: 'unchanged' });
    });

    it.each<[string, DocumentCommand, PrepareEditResult]>([
      ['a move already at the boundary', { ids: ['r1'], kind: 'front', type: 'move' }, { status: 'unchanged' }],
      [
        'a reorder that keeps the order',
        { orders: [{ orderedIds: ['r1', 'G', 'r5'], parentId: null, stack: 'raster' }], type: 'reorder' },
        { status: 'unchanged' },
      ],
    ])('reports %s as unchanged', (_label, command, result) => {
      expect(modelOf(projectWith(tree(), 'r1')).prepare(command)).toEqual(result);
    });

    it.each<[string, DocumentCommand, PrepareEditResult]>([
      ['a missing removal', { ids: ['ghost', 'r1'], type: 'remove' }, { ids: ['ghost'], status: 'missing' }],
      ['a locked removal', { ids: ['locked'], type: 'remove' }, { ids: ['locked'], status: 'locked' }],
      ['an empty removal', { ids: [], type: 'remove' }, { operation: 'remove nothing', status: 'unsupported' }],
      [
        'a clashing insert',
        { aboveId: null, nodes: [layer('r1')], type: 'insert' },
        { reason: 'id-exists', status: 'invalid-target', targetId: 'r1' },
      ],
      [
        'an insert clashing with a group id',
        { aboveId: null, nodes: [layer('G')], type: 'insert' },
        { reason: 'id-exists', status: 'invalid-target', targetId: 'G' },
      ],
      [
        'an empty insert',
        { aboveId: null, nodes: [], type: 'insert' },
        { operation: 'insert nothing', status: 'unsupported' },
      ],
      [
        'a missing duplicate source',
        { createId: () => 'x', ids: ['ghost'], type: 'duplicate' },
        { ids: ['ghost'], status: 'missing' },
      ],
      [
        'a duplicate onto an existing id',
        { createId: () => 'r1', ids: ['r5'], type: 'duplicate' },
        { reason: 'id-exists', status: 'invalid-target', targetId: 'r1' },
      ],
      ['an empty move', { ids: [], kind: 'front', type: 'move' }, { operation: 'move nothing', status: 'unsupported' }],
      [
        'a reorder missing a member',
        { orders: [{ orderedIds: ['r1'], parentId: null, stack: 'raster' }], type: 'reorder' },
        { reason: 'not-siblings', status: 'invalid-target', targetId: 'r1' },
      ],
      [
        'a reorder naming a node of another list',
        { orders: [{ orderedIds: ['r1', 'G', 'r5', 'r2'], parentId: null, stack: 'raster' }], type: 'reorder' },
        { reason: 'not-siblings', status: 'invalid-target', targetId: 'r2' },
      ],
      [
        'a reorder naming an unknown id',
        { orders: [{ orderedIds: ['r1', 'G', 'ghost'], parentId: null, stack: 'raster' }], type: 'reorder' },
        { ids: ['ghost'], status: 'missing' },
      ],
      [
        'a reorder under a leaf',
        { orders: [{ orderedIds: [], parentId: 'r1', stack: 'raster' }], type: 'reorder' },
        { reason: 'not-a-group', status: 'invalid-target', targetId: 'r1' },
      ],
      [
        'a reorder of one list twice',
        {
          orders: [
            { orderedIds: ['G', 'r1', 'r5'], parentId: null, stack: 'raster' },
            { orderedIds: ['r1', 'G', 'r5'], parentId: null, stack: 'raster' },
          ],
          type: 'reorder',
        },
        { operation: 'reorder one sibling list twice', status: 'unsupported' },
      ],
      [
        'a patch of nothing',
        { id: 'r1', patch: {}, type: 'patch' },
        { operation: 'patch nothing', status: 'unsupported' },
      ],
      ['a selection of a missing node', { id: 'ghost', type: 'select' }, { ids: ['ghost'], status: 'missing' }],
      ['an ungroup of a missing node', { ids: ['ghost'], type: 'ungroup' }, { ids: ['ghost'], status: 'missing' }],
      [
        'a translate of a missing node',
        { dx: 1, dy: 1, ids: ['ghost'], type: 'translate' },
        { ids: ['ghost'], status: 'missing' },
      ],
    ])('refuses %s', (_label, command, refusal) => {
      const project = projectWith([...tree(), layer('locked', 'raster', { isLocked: true })], 'r1');
      expect(modelOf(project).prepare(command)).toEqual(refusal);
    });

    it('stamps every prepared edit with the project id and edit revision it was built against', () => {
      const project = projectWith(tree(), 'r1');
      const result = createDocumentModel(project.canvas.document, { editRevision: 7, projectId: project.id }).prepare({
        ids: ['r1'],
        type: 'remove',
      });
      expect(result).toMatchObject({ edit: { expectedRevision: 7, projectId: project.id }, status: 'prepared' });
    });
  });

  describe('canMergeDown', () => {
    it('targets the raster sibling directly below under the same parent', () => {
      const project = projectWith([layer('a'), layer('b'), group('G', [layer('c'), layer('d')]), layer('e')], 'a');
      const model = modelOf(project);
      expect(model.canMergeDown('a')).toEqual({ lowerId: 'b', status: 'eligible', upperId: 'a' });
      expect(model.canMergeDown('b')).toEqual({ reason: 'no-layer-below', status: 'invalid-target', targetId: 'b' });
      expect(model.canMergeDown('c')).toEqual({ lowerId: 'd', status: 'eligible', upperId: 'c' });
      expect(model.canMergeDown('d')).toEqual({ reason: 'no-layer-below', status: 'invalid-target', targetId: 'd' });
      expect(model.canMergeDown('G')).toEqual({ actual: 'group', expected: ['raster'], status: 'wrong-type' });
    });

    it('reports a locked ancestor and a disabled ancestor', () => {
      const locked = projectWith([group('G', [layer('a'), layer('b')], { isLocked: true })], null);
      expect(modelOf(locked).canMergeDown('a')).toEqual({ ids: ['G'], status: 'locked' });
      const off = projectWith([group('G', [layer('a'), layer('b')], { isEnabled: false })], null);
      expect(modelOf(off).canMergeDown('a')).toEqual({
        reason: 'not-mergeable',
        status: 'invalid-target',
        targetId: 'a',
      });
    });

    it.each<[string, CanvasNodeContract[], string, object]>([
      ['the bottom layer', flat(), 'r3', { reason: 'no-layer-below', status: 'invalid-target', targetId: 'r3' }],
      ['a mask', flat(), 'i1', { actual: 'inpaint_mask', expected: ['raster'], status: 'wrong-type' }],
      ['a missing id', flat(), 'ghost', { ids: ['ghost'], status: 'missing' }],
      [
        'a locked lower layer',
        [layer('a'), layer('b', 'raster', { isLocked: true })],
        'a',
        { ids: ['b'], status: 'locked' },
      ],
      [
        'a disabled upper layer',
        [layer('a', 'raster', { isEnabled: false }), layer('b')],
        'a',
        { reason: 'not-mergeable', status: 'invalid-target', targetId: 'a' },
      ],
    ])('refuses %s', (_label, nodes, upperId, refusal) => {
      expect(modelOf(projectWith(nodes, null)).canMergeDown(upperId)).toEqual(refusal);
    });
  });
});

describe('document-level seam', () => {
  it('looks nodes, layers and leaves up through the same index the model uses', () => {
    const document = projectWith(tree(), 'r1').canvas.document;
    expect(lookupDocumentLayer(document, 'r3')).toBe(getDocumentLeaves(document).find((layer) => layer.id === 'r3'));
    expect(lookupDocumentLayer(document, 'G')).toBeNull();
    expect(lookupDocumentLeaf(document, 'r3')).toBe(compileDocumentLeaves(document).find((leaf) => leaf.id === 'r3'));
    expect(lookupDocumentLeaf(document, 'G')).toBeNull();
    expect(lookupDocumentLeaf(document, 'nope')).toBeNull();
  });

  it('finds the leaf sibling directly below, never crossing a parent or a group', () => {
    const document = projectWith(tree(), 'r1').canvas.document;
    expect(lookupLayerBelow(document, 'r2')).toBeNull();
    expect(lookupLayerBelow(document, 'r1')).toBeNull();
    expect(lookupLayerBelow(document, 'H')?.id).toBe('r4');
    expect(lookupLayerBelow(document, 'r4')).toBeNull();
    expect(lookupLayerBelow(projectWith([layer('a'), layer('b')], null).canvas.document, 'a')?.id).toBe('b');
  });

  it('shares the merge-down rule with the model', () => {
    const project = projectWith(tree(), 'r1');
    const model = modelOf(project);
    for (const entry of getDocumentIndex(project.canvas.document).nodes) {
      expect(mergeDownEligibility(project.canvas.document, entry.node.id)).toEqual(model.canMergeDown(entry.node.id));
    }
  });
});

describe('checkEditPostconditions', () => {
  const document = projectWith(tree(), 'r2').canvas.document;

  it.each([
    ['present ids', { ids: ['r1', 'G', 'r3'], kind: 'present' as const }, true],
    ['a missing present id', { ids: ['r1', 'ghost'], kind: 'present' as const }, false],
    ['absent ids', { ids: ['ghost'], kind: 'absent' as const }, true],
    ['a present absent id', { ids: ['r1'], kind: 'absent' as const }, false],
    [
      'the exact root order',
      { kind: 'sibling-order' as const, orderedIds: ['r1', 'G', 'r5'], parentId: null, stack: 'raster' as const },
      true,
    ],
    [
      'the exact group order',
      { kind: 'sibling-order' as const, orderedIds: ['r2', 'H', 'r4'], parentId: 'G', stack: 'raster' as const },
      true,
    ],
    [
      'a shorter order',
      { kind: 'sibling-order' as const, orderedIds: ['r1', 'G'], parentId: null, stack: 'raster' as const },
      false,
    ],
    [
      'a permuted order',
      { kind: 'sibling-order' as const, orderedIds: ['G', 'r1', 'r5'], parentId: null, stack: 'raster' as const },
      false,
    ],
    ['the selection', { id: 'r2', kind: 'selection' as const }, true],
    ['another selection', { id: 'r1', kind: 'selection' as const }, false],
    ['an applied patch', { id: 'r1', kind: 'patched' as const, patch: { name: 'r1', transform: { x: 0 } } }, true],
    ['an applied group patch', { id: 'G', kind: 'patched' as const, patch: { isEnabled: true, name: 'G' } }, true],
    ['an unapplied patch', { id: 'r1', kind: 'patched' as const, patch: { name: 'r1', transform: { x: 3 } } }, false],
    ['a leaf-only field on a group', { id: 'G', kind: 'patched' as const, patch: { opacity: 1 } }, false],
    ['a patch of a missing node', { id: 'ghost', kind: 'patched' as const, patch: { name: 'x' } }, false],
  ])('checks %s', (_label, postcondition, expected) => {
    expect(checkEditPostconditions(document, [postcondition])).toBe(expected);
  });
});

describe('lock coherence and limits', () => {
  /** raster: r1, L(locked)[r2, M[r3]], r4(locked) */
  const lockedTree = (): CanvasNodeContract[] => [
    layer('r1'),
    group('L', [layer('r2'), group('M', [layer('r3')])], { isLocked: true }),
    layer('r4', 'raster', { isLocked: true }),
  ];

  it('refuses content edits inside a locked group while name, enablement and lock flips stay open', () => {
    const model = modelOf(projectWith(lockedTree(), 'r2'));
    expect(model.prepare({ id: 'r2', patch: { transform: { x: 10 } }, type: 'patch' })).toEqual({
      ids: ['L'],
      status: 'locked',
    });
    expect(model.prepare({ id: 'r3', patch: { opacity: 0.5 }, type: 'patch' })).toEqual({
      ids: ['L'],
      status: 'locked',
    });
    expect(model.prepare({ id: 'r4', patch: { opacity: 0.5 }, type: 'patch' })).toEqual({
      ids: ['r4'],
      status: 'locked',
    });
    expect(model.prepare({ id: 'r2', source: { bitmap: null, type: 'paint' }, type: 'patch-source' })).toEqual({
      ids: ['L'],
      status: 'locked',
    });
    expect(model.prepare({ id: 'r2', patch: { name: 'renamed' }, type: 'patch' }).status).toBe('prepared');
    expect(model.prepare({ id: 'r3', patch: { isEnabled: false }, type: 'patch' }).status).toBe('prepared');
    expect(model.prepare({ id: 'L', patch: { isLocked: false }, type: 'patch' }).status).toBe('prepared');
  });

  it('refuses inserting into a locked group and every structural move that touches a locked subtree', () => {
    const model = modelOf(projectWith(lockedTree(), 'r2'));
    expect(model.prepare({ aboveId: 'r2', nodes: [layer('new')], type: 'insert' })).toEqual({
      ids: ['L'],
      status: 'locked',
    });
    expect(model.prepare({ aboveId: null, insideId: 'M', nodes: [layer('new')], type: 'insert' })).toEqual({
      ids: ['L'],
      status: 'locked',
    });
    expect(model.prepare({ ids: ['r2'], kind: 'back', type: 'move' })).toEqual({ ids: ['L'], status: 'locked' });
    expect(model.prepare({ ids: ['r4'], kind: 'front', type: 'move' })).toEqual({ ids: ['r4'], status: 'locked' });
    // A locked sibling may be passed; only the moving subtree has to be free.
    expect(model.prepare({ ids: ['r1'], kind: 'back', type: 'move' }).status).toBe('prepared');
    expect(model.prepare({ beforeId: null, ids: ['L'], parentId: null, type: 'reparent' })).toEqual({
      ids: ['L'],
      status: 'locked',
    });
    expect(model.prepare({ groupId: 'N', ids: ['r1', 'L'], name: 'n', type: 'group' })).toEqual({
      ids: ['L'],
      status: 'locked',
    });
    expect(model.prepare({ ids: ['L'], type: 'ungroup' })).toEqual({ ids: ['L'], status: 'locked' });
    expect(model.prepare({ aboveId: 'r1', nodes: [layer('new')], type: 'insert' }).status).toBe('prepared');
    expect(model.refusalFor({ ids: ['r1'], kind: 'front', type: 'move' })).toBeNull();
    expect(model.refusalFor({ ids: ['r4'], kind: 'front', type: 'move' })).toEqual({ ids: ['r4'], status: 'locked' });
  });

  it('counts every node of a multi-stack insert and a multi-root duplicate against the node limit', () => {
    const document = createLargeFlatDocument(CANVAS_MAX_NODE_COUNT - 1);
    const model = createDocumentModel(document, { editRevision: 0, projectId: 'project' });
    expect(model.prepare({ aboveId: null, nodes: [layer('x'), layer('y', 'control')], type: 'insert' })).toMatchObject({
      reason: 'node-limit',
      status: 'invalid-target',
    });
    expect(model.prepare({ aboveId: null, nodes: [layer('x')], type: 'insert' }).status).toBe('prepared');
    let created = 0;
    const createId = () => `dup${created++}`;
    expect(model.prepare({ createId, ids: ['l0', 'l1'], type: 'duplicate' })).toMatchObject({ reason: 'node-limit' });
    expect(model.prepare({ createId, ids: ['l0'], type: 'duplicate' }).status).toBe('prepared');
  });

  it('inserts an empty group into the stack it names and refuses one without a stack', () => {
    expect(
      modelOf(projectWith(tree(), 'c1')).prepare({ aboveId: 'c1', nodes: [group('E')], type: 'insert' })
    ).toMatchObject({
      status: 'unsupported',
    });
    const { after } = roundTrip(projectWith(tree(), 'c1'), {
      aboveId: 'c1',
      nodes: [group('E')],
      stack: 'control',
      type: 'insert',
    });
    expect(after.canvas.document.stacks.control.map((node) => node.id)).toEqual(['E', 'c1']);
  });

  it('reparents leaves from different parents in one edit and restores each run on undo', () => {
    const { after } = roundTrip(projectWith(tree(), 'r1'), {
      beforeId: null,
      ids: ['r1', 'r3', 'r5'],
      parentId: 'G',
      type: 'reparent',
    });
    expect(outline(after.canvas.document)).toEqual(['i1', 'c1', 'G', '  r2', '  H', '  r4', '  r1', '  r3', '  r5']);
  });

  it('dissolves sibling groups in one edit', () => {
    const { after, edit } = roundTrip(
      projectWith([group('A', [layer('r1'), layer('r2')]), group('B', [layer('r3')])], 'r3'),
      {
        ids: ['A', 'B'],
        type: 'ungroup',
      }
    );
    expect(outline(after.canvas.document)).toEqual(['r1', 'r2', 'r3']);
    expect(edit.selectionAfter).toBe('r3');
  });
});
