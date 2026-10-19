import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';

import {
  getDocumentIndex,
  getDocumentIndexBuildCount,
  getDocumentIndexDerivationCount,
  getDocumentIndexMaterializationCount,
  getDocumentIndexVisitCount,
  getValueEditVisitCount,
  resetDocumentIndexBuildCount,
} from '@workbench/canvas-engine/document/documentIndex';
import {
  createDocumentMirror,
  getForestDiffCount,
  resetForestDiffCount,
} from '@workbench/canvas-engine/document/documentMirror';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { DocumentCommand, PreparedDocumentEdit } from './documentCommands';

import { createLargeFlatDocument, createLargeTreeDocument } from './documentFixtures.testStub';
import {
  compileDocumentLeaves,
  compileDocumentNodes,
  createDocumentModel,
  lookupDocumentLeaf,
  lookupDocumentNodeState,
  getDocumentModelDiagnostics,
  resetDocumentModelDiagnostics,
} from './documentModel';
import { ALL_OVERLAY_STACKS_SHOWN, planScreenComposition } from './screenComposition';

const NODE_COUNT = 2_000;
const context = { editRevision: 0, projectId: 'budget' };

const resetCounters = (): void => {
  resetDocumentModelDiagnostics();
  resetDocumentIndexBuildCount();
  resetForestDiffCount();
};

const counters = () => ({
  ...getDocumentModelDiagnostics(),
  entriesVisited: getDocumentIndexVisitCount(),
  forestDiffs: getForestDiffCount(),
  forestNodesVisited: getValueEditVisitCount(),
  indexBuilds: getDocumentIndexBuildCount(),
  indexDerivations: getDocumentIndexDerivationCount(),
  nodesMaterialized: getDocumentIndexMaterializationCount(),
});

/** Runs the reducer over a fixture so the next document is the one production consumers see. */
const reduce = (document: CanvasDocumentContractV3, command: DocumentCommand) => {
  const initial = createInitialWorkbenchState().projects[0]!;
  const project = applyCanvasProjectMutation(initial, { document, type: 'replaceCanvasDocument' });
  const accepted = project.canvas.document;
  const model = createDocumentModel(accepted, { editRevision: 0, projectId: project.id });
  const result = model.prepare(command);
  if (result.status !== 'prepared') {
    throw new Error(`expected a prepared edit, got ${result.status}`);
  }
  return { edit: result.edit, model, next: applyCanvasProjectMutation(project, result.edit.forward).canvas.document };
};

type Annotate = (message: string) => unknown;

/** Timing is informational: it is annotated on the test, never asserted. */
const timed = <T>(annotate: Annotate, label: string, run: () => T): T => {
  const start = performance.now();
  const value = run();
  annotate(`${label}: ${(performance.now() - start).toFixed(2)}ms`);
  return value;
};

describe(`document model budgets over ${NODE_COUNT} nodes`, () => {
  beforeEach(resetCounters);

  it('builds one index per forest identity and none for selection-only changes', ({ annotate }) => {
    const fixture = createLargeFlatDocument(NODE_COUNT);
    resetCounters();
    const project = applyCanvasProjectMutation(createInitialWorkbenchState().projects[0]!, {
      document: fixture,
      type: 'replaceCanvasDocument',
    });
    const document = project.canvas.document;
    expect(counters().indexBuilds).toBe(1);
    const model = timed(annotate, 'model construction', () => createDocumentModel(document, context));
    createDocumentModel(document, context);
    model.getLayer('l1999');
    model.getStack('control');
    expect(model.prepare({ id: 'l7', type: 'select' }).status).toBe('prepared');
    expect(model.prepare({ id: 'l0', type: 'select' })).toEqual({ status: 'unchanged' });
    const leaves = compileDocumentLeaves(document);

    const reselected = applyCanvasProjectMutation(project, { id: 'l7', type: 'setCanvasSelectedLayer' }).canvas
      .document;
    expect(reselected).not.toBe(document);
    expect(reselected.stacks).toBe(document.stacks);
    expect(createDocumentModel(reselected, context).compileLeaves()).toBe(leaves);
    expect(counters()).toMatchObject({ indexBuilds: 1, leafCompilations: 1 });
  });

  it('compiles leaves once per forest and plans the screen from them', ({ annotate }) => {
    const document = createLargeTreeDocument(NODE_COUNT);
    const leafCount = getDocumentIndex(document).leaves.length;
    const leaves = timed(annotate, 'leaf compilation', () => compileDocumentLeaves(document));
    expect(compileDocumentLeaves(document)).toBe(leaves);
    expect(createDocumentModel(document, context).compileLeaves()).toBe(leaves);
    expect(counters()).toMatchObject({ indexBuilds: 1, leafCompilations: 1, leavesCompiled: leafCount });

    const plan = timed(annotate, 'screen plan', () =>
      planScreenComposition(leaves, { isolationLayerId: null, showOverlayStacks: ALL_OVERLAY_STACKS_SHOWN })
    );
    expect(plan.leaves).toHaveLength(leafCount);
    expect(plan.leaves.at(-1)?.stack).toBe('inpaint_mask');
  });

  it('recompiles only the patched leaf after a reducer edit inside a group', ({ annotate }) => {
    const { model, next } = reduce(createLargeTreeDocument(NODE_COUNT), {
      id: 'l5',
      patch: { opacity: 0.5 },
      type: 'patch',
    });
    const before = model.compileLeaves();
    resetCounters();
    const after = timed(annotate, 'representative edit recompile', () => compileDocumentLeaves(next));
    expect(counters()).toMatchObject({ indexBuilds: 0, leafCompilations: 1, leavesCompiled: 1 });
    after.forEach((leaf, index) => {
      if (leaf.id === 'l5') {
        expect(leaf).not.toBe(before[index]);
        expect(leaf.layer.opacity).toBe(0.5);
      } else {
        expect(leaf).toBe(before[index]);
      }
    });
  });

  it('recompiles zero leaves for a group rename and exactly the descendants for a group flag', () => {
    const renamed = reduce(createLargeTreeDocument(NODE_COUNT), { id: 'g0', patch: { name: 'Folder' }, type: 'patch' });
    const before = renamed.model.compileLeaves();
    resetCounters();
    expect(compileDocumentLeaves(renamed.next)).toEqual(before);
    expect(counters()).toMatchObject({ indexBuilds: 0, leavesCompiled: 0 });

    const gated = reduce(createLargeTreeDocument(NODE_COUNT), {
      type: 'set-enabled',
      updates: [{ id: 'g0', isEnabled: false }],
    });
    const descendants = getDocumentIndex(gated.next).leaves.filter((leaf) =>
      getDocumentIndex(gated.next).byId.get(leaf.id)!.path.includes('g0')
    );
    gated.model.compileLeaves();
    resetCounters();
    const after = compileDocumentLeaves(gated.next);
    expect(counters().leavesCompiled).toBe(descendants.length);
    expect(after.filter((leaf) => !leaf.contributionEnabled).map((leaf) => leaf.id)).toEqual(
      descendants.map((leaf) => leaf.id)
    );
  });

  it('preserves every leaf identity across a sibling reorder and a reparent', () => {
    const moved = reduce(createLargeTreeDocument(NODE_COUNT), { ids: ['l0'], kind: 'back', type: 'move' });
    const before = moved.model.compileLeaves();
    resetCounters();
    const reordered = compileDocumentLeaves(moved.next);
    expect(reordered).toHaveLength(before.length);
    expect(reordered.every((leaf) => before.includes(leaf)) && before.every((leaf) => reordered.includes(leaf))).toBe(
      true
    );
    expect(reordered.map((leaf) => leaf.id)).not.toEqual(before.map((leaf) => leaf.id));
    expect(counters().leavesCompiled).toBe(0);

    const tree = createLargeTreeDocument(NODE_COUNT);
    const entries = getDocumentIndex(tree).nodes.filter((entry) => entry.stack === 'raster');
    const targetGroup = entries.find((entry) => entry.node.type === 'group')!.node.id;
    const rootLeaf = entries.find((entry) => entry.node.type !== 'group' && !entry.path.includes(targetGroup))!.node.id;
    const reparented = reduce(tree, { beforeId: null, ids: [rootLeaf], parentId: targetGroup, type: 'reparent' });
    const untouched = reparented.model.compileLeaves();
    resetCounters();
    const after = compileDocumentLeaves(reparented.next);
    expect(counters().leavesCompiled).toBe(1);
    expect(after).toHaveLength(untouched.length);
    expect(after.filter((leaf) => leaf.id !== rootLeaf).every((leaf) => untouched.includes(leaf))).toBe(true);
    expect(after.find((leaf) => leaf.id === rootLeaf)).not.toBe(untouched.find((leaf) => leaf.id === rootLeaf));
  });

  it.each<[string, DocumentCommand, { ids: number; stacks: number; builds: number }]>([
    ['select', { id: 'l9', type: 'select' }, { builds: 1, ids: 0, stacks: 0 }],
    ['lock', { type: 'set-locked', updates: [{ id: 'l1', isLocked: true }] }, { builds: 1, ids: 1, stacks: 1 }],
    [
      'enable',
      {
        type: 'set-enabled',
        updates: Array.from({ length: 10 }, (_, index) => ({ id: `l${index}`, isEnabled: false })),
      },
      { builds: 1, ids: 10, stacks: 4 },
    ],
    ['hide', { type: 'set-hidden', updates: [{ id: 'l1', isHidden: true }] }, { builds: 1, ids: 1, stacks: 1 }],
    ['move', { ids: ['l8', 'l12', 'l16'], kind: 'forward', type: 'move' }, { builds: 1, ids: 4, stacks: 1 }],
    // Removal repairs the selection over the projected forest, which costs one more index.
    ['remove', { ids: ['l1', 'l2'], type: 'remove' }, { builds: 2, ids: 2, stacks: 2 }],
  ])('prepares a %s edit with exact touched ids and stacks and no leaf compilation', (_label, command, expected) => {
    const model = createDocumentModel(createLargeFlatDocument(NODE_COUNT), context);
    const result = model.prepare(command);
    expect(result.status).toBe('prepared');
    const edit = (result as { edit: PreparedDocumentEdit }).edit;
    expect(edit.touchedIds).toHaveLength(expected.ids);
    expect(edit.touchedStacks).toHaveLength(expected.stacks);
    expect(edit.rasterWork).toBeNull();
    expect(counters()).toMatchObject({ indexBuilds: expected.builds, leafCompilations: 0 });
  });

  it('prepares group, ungroup and reparent edits on the tree without compiling leaves', () => {
    const model = createDocumentModel(createLargeTreeDocument(NODE_COUNT), context);
    // l0 sits inside g0, so the selection folds into its ancestor and groups as one node.
    expect(model.prepare({ groupId: 'new', ids: ['g0', 'l0'], name: 'x', type: 'group' }).status).toBe('prepared');
    expect(model.prepare({ ids: ['g0'], type: 'ungroup' }).status).toBe('prepared');
    expect(model.prepare({ beforeId: null, ids: ['l0'], parentId: 'g0', type: 'reparent' }).status).toBe('prepared');
    expect(counters()).toMatchObject({ indexBuilds: 1, leafCompilations: 0 });
  });
});

describe(`value edits over ${NODE_COUNT} nodes`, () => {
  beforeEach(resetCounters);

  it('derives the index for a geometry edit instead of rebuilding it, keeping every other leaf and node', ({
    annotate,
  }) => {
    const fixture = createLargeTreeDocument(NODE_COUNT);
    resetCounters();
    const { model, next } = reduce(fixture, { id: 'l5', patch: { transform: { x: 40 } }, type: 'patch' });
    const leavesBefore = model.compileLeaves();
    const nodesBefore = compileDocumentNodes(model.document);
    // One build for the accepted fixture, one derivation for the edit, and nothing on lookup.
    const index = timed(annotate, 'derived index lookup', () => getDocumentIndex(next));
    expect(counters()).toMatchObject({ indexBuilds: 1, indexDerivations: 1 });
    expect(index.byId.get('l5')!.node).not.toBe(model.getLayer('l5'));
    expect(model.getLayer('l5')).not.toBeNull();
    expect((index.byId.get('l5')!.node as { transform: { x: number } }).transform).toMatchObject({ x: 40 });
    expect(index.byId.get('l6')).toBe(model.getEntry('l6'));

    resetCounters();
    const leavesAfter = compileDocumentLeaves(next);
    const nodesAfter = compileDocumentNodes(next);
    // The leaf and the ancestors rebuilt along its path are new contract objects; nothing else is.
    expect(counters()).toMatchObject({
      indexBuilds: 0,
      indexDerivations: 0,
      leavesCompiled: 1,
      nodesCompiled: 1 + index.byId.get('l5')!.path.length,
    });
    const path = new Set(index.byId.get('l5')!.path);
    expect(leavesAfter.filter((leaf) => leaf.id !== 'l5').every((leaf) => leavesBefore.includes(leaf))).toBe(true);
    expect(
      nodesAfter.filter((node) => node.id !== 'l5' && !path.has(node.id)).every((node) => nodesBefore.includes(node))
    ).toBe(true);
  });

  it('re-derives only the subtree under a group whose flags change', () => {
    resetCounters();
    const { model, next } = reduce(createLargeTreeDocument(NODE_COUNT), {
      type: 'set-locked',
      updates: [{ id: 'g0', isLocked: true }],
    });
    const before = getDocumentIndex(model.document);
    const after = getDocumentIndex(next);
    expect(counters()).toMatchObject({ indexBuilds: 1, indexDerivations: 1 });
    const subtree = before.nodes.filter((entry) => entry.path.includes('g0'));
    expect(subtree.length).toBeGreaterThan(0);
    expect(subtree.every((entry) => after.byId.get(entry.node.id)!.ancestorsLocked)).toBe(true);
    const untouched = before.nodes.filter((entry) => !entry.path.includes('g0') && entry.node.id !== 'g0');
    expect(untouched.every((entry) => after.byId.get(entry.node.id) === entry)).toBe(true);
    expect(after.leaves).toBe(before.leaves);
  });

  it('keeps the leaves array identity when a group is renamed', () => {
    const { model, next } = reduce(createLargeTreeDocument(NODE_COUNT), {
      id: 'g0',
      patch: { name: 'renamed' },
      type: 'patch',
    });
    const before = compileDocumentLeaves(model.document);
    resetCounters();
    expect(getDocumentIndex(next).leaves).toBe(getDocumentIndex(model.document).leaves);
    compileDocumentLeaves(next).forEach((leaf, position) => expect(leaf).toBe(before[position]));
    expect(counters()).toMatchObject({ indexBuilds: 0, indexDerivations: 0, leavesCompiled: 0 });
  });
});

describe('value edits at the 10,000-node limit', () => {
  const LIMIT_COUNT = 10_000;
  beforeEach(resetCounters);

  it('costs one entry per changed path, and nothing else, per value edit', () => {
    const fixture = createLargeTreeDocument(LIMIT_COUNT);
    const initial = createInitialWorkbenchState().projects[0]!;
    const project = applyCanvasProjectMutation(initial, { document: fixture, type: 'replaceCanvasDocument' });
    const before = project.canvas.document;
    compileDocumentNodes(before);
    compileDocumentLeaves(before);
    lookupDocumentLeaf(before, 'l5');
    const depth = getDocumentIndex(before).byId.get('l5')!.path.length;
    resetCounters();

    const next = applyCanvasProjectMutation(project, {
      id: 'l5',
      patch: { transform: { x: 40 } },
      type: 'updateCanvasLayer',
    }).canvas.document;
    expect(counters()).toMatchObject({
      entriesVisited: depth + 1,
      forestNodesVisited: depth + 1,
      indexBuilds: 0,
      indexDerivations: 1,
    });
    expect(lookupDocumentLeaf(next, 'l5')?.layer.transform.x).toBe(40);
    expect(lookupDocumentNodeState(next, 'l5')?.node).toBe(getDocumentIndex(next).byId.get('l5')!.node);
    expect(lookupDocumentNodeState(next, 'l6')).toBe(lookupDocumentNodeState(before, 'l6'));
    expect(counters()).toMatchObject({ leavesCompiled: 1, nodesCompiled: depth + 1, nodesMaterialized: 0 });
    // The ancestors compiled again kept their counts without walking their subtrees.
    const parentId = getDocumentIndex(next).byId.get('l5')!.parentId!;
    expect(lookupDocumentNodeState(next, parentId)?.leafCount).toBe(
      lookupDocumentNodeState(before, parentId)?.leafCount
    );
  });

  it('costs the subtree, and only the subtree, for a group flag flip', () => {
    const fixture = createLargeTreeDocument(LIMIT_COUNT);
    const initial = createInitialWorkbenchState().projects[0]!;
    const project = applyCanvasProjectMutation(initial, { document: fixture, type: 'replaceCanvasDocument' });
    const before = project.canvas.document;
    const subtree = getDocumentIndex(before).nodes.filter((entry) => entry.path.includes('g0')).length;
    compileDocumentNodes(before);
    resetCounters();
    const next = applyCanvasProjectMutation(project, {
      type: 'setCanvasLayersEnabled',
      updates: [{ id: 'g0', isEnabled: false }],
    }).canvas.document;
    expect(counters()).toMatchObject({
      entriesVisited: subtree + 1,
      forestNodesVisited: 1,
      indexBuilds: 0,
      indexDerivations: 1,
    });
    expect(getDocumentIndex(next).leaves).toBe(getDocumentIndex(before).leaves);
    compileDocumentNodes(next);
    expect(counters()).toMatchObject({ nodesCompiled: subtree + 1, nodesMaterialized: 0 });
  });

  it('flattens the derivation chain so lookups stay bounded and compiles stay incremental across many edits', () => {
    const fixture = createLargeTreeDocument(LIMIT_COUNT);
    const initial = createInitialWorkbenchState().projects[0]!;
    let project = applyCanvasProjectMutation(initial, { document: fixture, type: 'replaceCanvasDocument' });
    compileDocumentNodes(project.canvas.document);
    resetCounters();
    let deepest = 0;
    let compiledMost = 0;
    for (let step = 0; step < 20; step += 1) {
      const id = `l${step}`;
      const depth = getDocumentIndex(project.canvas.document).byId.get(id)!.path.length;
      project = applyCanvasProjectMutation(project, { id, patch: { opacity: 0.5 }, type: 'updateCanvasLayer' });
      const before = counters().nodesCompiled;
      compileDocumentNodes(project.canvas.document);
      compiledMost = Math.max(compiledMost, (counters().nodesCompiled - before) / (depth + 1));
      let chain = 0;
      for (
        let index = getDocumentIndex(project.canvas.document);
        index.derivedFrom;
        index = index.derivedFrom.previous
      ) {
        chain += 1;
      }
      deepest = Math.max(deepest, chain);
    }
    expect(deepest).toBeLessThanOrEqual(9);
    // The flattening step recompiles what the chain touched, never the document.
    expect(compiledMost).toBeLessThanOrEqual(9);
    expect(counters()).toMatchObject({ indexBuilds: 0, nodesMaterialized: 0 });
    expect(getDocumentIndex(project.canvas.document).byId.get('l19')!.node).toMatchObject({ opacity: 0.5 });
    expect(getDocumentIndex(project.canvas.document).byId.size).toBe(LIMIT_COUNT);
  });
});

describe('reducer → mirror value edits at the 10,000-node limit', () => {
  beforeEach(resetCounters);

  it.for([
    ['flat', createLargeFlatDocument],
    ['tree', createLargeTreeDocument],
  ] as const)('touches only the edited path end to end on a %s document', ([, create], { annotate }) => {
    let project = applyCanvasProjectMutation(createInitialWorkbenchState().projects[0]!, {
      document: create(10_000),
      type: 'replaceCanvasDocument',
    });
    const onLayersChanged = vi.fn();
    const mirror = createDocumentMirror(
      { getCanvasState: () => project.canvas, subscribe: () => () => undefined },
      {
        onBboxChanged: () => undefined,
        onDocumentReplaced: () => undefined,
        onLayerOrderChanged: () => undefined,
        onLayersChanged,
        onStagingChanged: () => undefined,
      }
    );
    const depth = getDocumentIndex(project.canvas.document).byId.get('l0')!.path.length;
    resetCounters();

    timed(annotate, 'reducer + mirror', () => {
      project = applyCanvasProjectMutation(project, { id: 'l0', patch: { opacity: 0.5 }, type: 'updateCanvasLayer' });
      mirror.refresh();
    });
    expect(onLayersChanged).toHaveBeenCalledWith(['l0'], [], false);
    expect(counters()).toMatchObject({
      forestDiffs: 0,
      forestNodesVisited: depth + 1,
      indexBuilds: 0,
      indexDerivations: 1,
      nodesMaterialized: 0,
    });
    mirror.dispose();
  });
});

describe('eligibility without materialization', () => {
  beforeEach(resetCounters);

  it('answers reparent, group, ungroup and remove eligibility without building an edit', () => {
    const document = createLargeTreeDocument(NODE_COUNT);
    const model = createDocumentModel(document, context);
    const index = getDocumentIndex(document);
    const stack = index.byId.get('l0')!.stack;
    const groupId = index.nodes.find((entry) => entry.node.type === 'group' && entry.stack === stack)!.node.id;
    resetCounters();
    expect(model.refusalFor({ beforeId: null, ids: ['l0'], parentId: groupId, type: 'reparent' })).toBeNull();
    expect(model.refusalFor({ beforeId: null, ids: [groupId], parentId: groupId, type: 'reparent' })).toMatchObject({
      reason: 'cycle',
    });
    expect(model.refusalFor({ groupId: '\0probe', ids: ['l0', 'l1'], name: '', type: 'group' })).toBeDefined();
    expect(model.refusalFor({ ids: [groupId], type: 'ungroup' })).toBeNull();
    expect(model.refusalFor({ ids: ['l0'], type: 'remove' })).toBeNull();
    expect(counters().editsMaterialized).toBe(0);
    expect(model.prepare({ ids: ['l0'], type: 'remove' }).status).toBe('prepared');
    expect(counters().editsMaterialized).toBe(1);
  });
});
