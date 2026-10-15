import type { CanvasDocumentContractV3, CanvasNodeContract } from '@workbench/canvas-engine/contracts';
import type { Project } from '@workbench/projectContracts';

import { CANVAS_MAX_NODE_COUNT, CANVAS_MAX_NODE_DEPTH, LAYER_STACK_ORDER } from '@workbench/canvas-engine/contracts';
import { getDocumentIndex } from '@workbench/canvas-engine/document/documentIndex';
import { collectSubtree, isGroupNode } from '@workbench/canvas-engine/document/documentTree';
import { haveSameStructure } from '@workbench/canvas-engine/document/layerStacks';
import { createEmptyCanvasDocument, normalizeCanvasDocumentContract } from '@workbench/canvasMigration';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { describe, expect, it } from 'vitest';

import type { DocumentCommand } from './documentCommands';

import { groupContract, layerContract, stacksFrom } from './documentFixtures.testStub';
import { createDocumentModel } from './documentModel';
import { checkEditPostconditions } from './postconditions';

/** Deterministic 32-bit generator so a failing seed reproduces exactly. */
const rng = (seed: number) => {
  let state = seed >>> 0;
  return (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

const pick = <T>(random: () => number, items: readonly T[]): T => items[Math.floor(random() * items.length)]!;
const sample = <T>(random: () => number, items: readonly T[], max: number): T[] => {
  const pool = [...items];
  const out: T[] = [];
  const count = 1 + Math.floor(random() * Math.min(max, pool.length));
  while (out.length < count && pool.length > 0) {
    out.push(pool.splice(Math.floor(random() * pool.length), 1)[0]!);
  }
  return out;
};

let created = 0;
const nextId = (): string => `n${created++}`;

/** A random forest: every stack gets a few leaves, some wrapped in groups up to `depth`. */
const randomTree = (random: () => number, depth: number, budget: { left: number }): CanvasNodeContract[] => {
  const nodes: CanvasNodeContract[] = [];
  const count = 1 + Math.floor(random() * 3);
  for (let index = 0; index < count && budget.left > 0; index += 1) {
    budget.left -= 1;
    if (depth > 0 && random() < 0.55) {
      nodes.push(
        groupContract(nextId(), randomTree(random, depth - 1, budget), {
          isEnabled: random() < 0.85,
          isLocked: random() < 0.15,
        })
      );
    } else {
      nodes.push(layerContract(nextId(), 'raster', { isEnabled: random() < 0.85, isLocked: random() < 0.1 }));
    }
  }
  return nodes;
};

const randomDocument = (random: () => number, depth: number): CanvasDocumentContractV3 => {
  const stacks = stacksFrom([]);
  for (const stack of LAYER_STACK_ORDER) {
    const budget = { left: 4 + Math.floor(random() * 8) };
    stacks[stack] = randomTree(random, depth, budget).map((node) => retype(node, stack, random));
  }
  const ids = Object.values(stacks).flatMap((roots) => roots.flatMap((root) => collectSubtree(root).map((n) => n.id)));
  return { ...createEmptyCanvasDocument(), selectedLayerId: ids[0] ?? null, stacks };
};

/** Rebuilds a raster fixture subtree for `stack`; overlay groups may start display-hidden. */
const retype = (
  node: CanvasNodeContract,
  stack: (typeof LAYER_STACK_ORDER)[number],
  random: () => number
): CanvasNodeContract =>
  isGroupNode(node)
    ? {
        ...node,
        children: node.children.map((child) => retype(child, stack, random)),
        ...(stack !== 'raster' && random() < 0.2 ? { isHidden: true } : {}),
      }
    : { ...layerContract(node.id, stack, { isEnabled: node.isEnabled, isLocked: node.isLocked }) };

const randomCommand = (random: () => number, document: CanvasDocumentContractV3): DocumentCommand => {
  const index = getDocumentIndex(document);
  const ids = index.nodes.map((entry) => entry.node.id);
  const groups = index.nodes.filter((entry) => isGroupNode(entry.node));
  const groupIds = groups.map((entry) => entry.node.id);
  const overlayIds = index.nodes.filter((entry) => entry.stack !== 'raster').map((entry) => entry.node.id);
  const kind = pick(random, [
    'insert',
    'insert',
    'remove',
    'duplicate',
    'duplicate',
    'move',
    'reparent',
    'reparent',
    'reparent',
    'group',
    'group',
    'ungroup',
    'ungroup',
    'flags',
    'hidden',
    'patch',
    'translate',
  ] as const);
  switch (kind) {
    case 'insert': {
      const anchor = pick(random, [...ids, null]);
      const stack = anchor ? index.byId.get(anchor)!.stack : pick(random, LAYER_STACK_ORDER);
      const nodes: CanvasNodeContract[] = [];
      const roll = random();
      if (roll < 0.25) {
        nodes.push(groupContract(nextId()));
      } else if (roll < 0.5) {
        nodes.push(
          groupContract(nextId(), [
            layerContract(nextId(), stack),
            groupContract(nextId(), [layerContract(nextId(), stack)]),
          ])
        );
      } else if (roll < 0.7) {
        nodes.push(layerContract(nextId(), stack), layerContract(nextId(), stack));
      } else {
        nodes.push(layerContract(nextId(), stack));
      }
      const inside = groups.filter((entry) => entry.stack === stack);
      return random() < 0.5 && inside.length > 0
        ? {
            aboveId: null,
            insideId: pick(random, inside).node.id,
            nodes,
            selectId: random() < 0.3 ? null : undefined,
            stack,
            type: 'insert',
          }
        : { aboveId: anchor, nodes, stack, type: 'insert' };
    }
    case 'remove':
      return { ids: sample(random, ids, 3), type: 'remove' };
    case 'duplicate': {
      // Half the time pick a group and something inside it, so nested selections fold.
      const nested = groups.length > 0 && random() < 0.5 ? pick(random, groups) : null;
      const inner = nested ? index.nodes.filter((entry) => entry.path.includes(nested.node.id)) : [];
      const chosen =
        nested && inner.length > 0 ? [nested.node.id, pick(random, inner).node.id] : sample(random, ids, 2);
      return { createId: nextId, ids: chosen, type: 'duplicate' };
    }
    case 'move':
      return {
        ids: sample(random, ids, 3),
        kind: pick(random, ['front', 'forward', 'backward', 'back'] as const),
        type: 'move',
      };
    case 'reparent': {
      // Parent and moving ids come from one stack, so the stack rule is not what usually refuses.
      const stack = pick(random, LAYER_STACK_ORDER);
      const inStack = index.nodes.filter((entry) => entry.stack === stack);
      if (inStack.length === 0) {
        return { ids: sample(random, ids, 1), kind: 'front', type: 'move' };
      }
      const parents = inStack.filter((entry) => isGroupNode(entry.node));
      const parentId = random() < 0.7 && parents.length > 0 ? pick(random, parents).node.id : null;
      const siblings = inStack.filter((entry) => entry.parentId === parentId).map((entry) => entry.node.id);
      return {
        beforeId: random() < 0.5 && siblings.length > 0 ? pick(random, siblings) : null,
        ids: sample(
          random,
          inStack.map((entry) => entry.node.id),
          3
        ),
        parentId,
        type: 'reparent',
      };
    }
    case 'group': {
      // Siblings most of the time, so the depth and lock rules get exercised, not just not-siblings.
      const anchor = pick(random, index.nodes);
      const siblings = index.nodes
        .filter((entry) => entry.stack === anchor.stack && entry.parentId === anchor.parentId)
        .map((entry) => entry.node.id);
      return {
        groupId: nextId(),
        ids: random() < 0.8 ? sample(random, siblings, 3) : sample(random, ids, 3),
        name: 'fuzz',
        type: 'group',
      };
    }
    case 'ungroup':
      return { ids: groupIds.length > 0 ? sample(random, groupIds, 3) : [ids[0]!], type: 'ungroup' };
    case 'flags': {
      const type = pick(random, ['set-enabled', 'set-locked'] as const);
      const flag = random() < 0.5;
      const updates = sample(random, ids, 3);
      return type === 'set-enabled'
        ? { type, updates: updates.map((id) => ({ id, isEnabled: flag })) }
        : { type, updates: updates.map((id) => ({ id, isLocked: flag })) };
    }
    case 'hidden': {
      const targets = overlayIds.length > 0 ? sample(random, overlayIds, 3) : sample(random, ids, 1);
      const isHidden = random() < 0.5;
      return { type: 'set-hidden', updates: targets.map((id) => ({ id, isHidden })) };
    }
    case 'patch': {
      const id = pick(random, ids);
      const roll = random();
      const patch =
        roll < 0.3
          ? { name: `n${Math.floor(random() * 100)}` }
          : roll < 0.55
            ? { opacity: random() }
            : roll < 0.8
              ? { transform: { x: Math.round(random() * 50) } }
              : // Clears exercise the explicit-undefined inverse path.
                { colorLabel: random() < 0.3 ? undefined : random() < 0.5 ? ('red' as const) : ('blue' as const) };
      return { id, patch, type: 'patch' };
    }
    case 'translate':
      return {
        dx: Math.round(random() * 20) - 10,
        dy: Math.round(random() * 20) - 10,
        ids: sample(random, ids, 2),
        type: 'translate',
      };
  }
};

/** Every invariant the definition of done names for a document at rest. */
const assertInvariants = (document: CanvasDocumentContractV3): void => {
  const index = getDocumentIndex(document);
  expect(index.byId.size).toBe(index.nodes.length);
  expect(index.maxDepth).toBeLessThanOrEqual(CANVAS_MAX_NODE_DEPTH);
  for (const entry of index.nodes) {
    if (!isGroupNode(entry.node)) {
      expect(entry.node.type).toBe(entry.stack);
    }
    expect(entry.path.includes(entry.node.id)).toBe(false);
  }
  if (document.selectedLayerId !== null) {
    expect(index.byId.has(document.selectedLayerId)).toBe(true);
  }
  expect(normalizeCanvasDocumentContract(document)).not.toBeNull();
};

const projectFor = (document: CanvasDocumentContractV3): Project =>
  applyCanvasProjectMutation(createInitialWorkbenchState().projects[0]!, { document, type: 'replaceCanvasDocument' });

const outcomes = { locked: 0, nestedDuplicate: 0, nestedUngroup: 0, reparentInside: 0 };

describe('document model fuzzing', () => {
  it.each(Array.from({ length: 160 }, (_, seed) => seed + 1))(
    'keeps every invariant and round-trips undo through random edits (seed %i)',
    (seed) => {
      const random = rng(seed);
      created = 0;
      let project = projectFor(randomDocument(random, seed % 9));
      assertInvariants(project.canvas.document);
      let applied = 0;
      for (let step = 0; step < 16; step += 1) {
        const before = project;
        const model = createDocumentModel(before.canvas.document, { editRevision: step, projectId: before.id });
        const command = randomCommand(random, before.canvas.document);
        const result = model.prepare(command);
        expect(model.refusalFor(command)).toEqual(
          result.status === 'prepared' || result.status === 'unchanged' ? null : result
        );
        if (result.status === 'locked') {
          outcomes.locked += 1;
        }
        if (result.status !== 'prepared') {
          continue;
        }
        const preIndex = getDocumentIndex(before.canvas.document);
        const nestedSelection = (selected: readonly string[]): boolean =>
          selected.some((id) => preIndex.byId.get(id)!.path.some((ancestor) => selected.includes(ancestor)));
        if (command.type === 'duplicate' && nestedSelection(command.ids)) {
          outcomes.nestedDuplicate += 1;
        }
        if (command.type === 'ungroup' && nestedSelection(command.ids)) {
          outcomes.nestedUngroup += 1;
        }
        if (command.type === 'reparent' && command.parentId !== null) {
          outcomes.reparentInside += 1;
        }
        const after = applyCanvasProjectMutation(before, result.edit.forward);
        expect(after.canvas.document, `${JSON.stringify(command)} was prepared but the reducer refused it`).not.toBe(
          before.canvas.document
        );
        assertInvariants(after.canvas.document);
        expect(checkEditPostconditions(after.canvas.document, result.edit.postconditions)).toBe(true);
        expect(after.canvas.document.selectedLayerId).toBe(result.edit.selectionAfter);
        const restored = applyCanvasProjectMutation(after, result.edit.inverse).canvas.document;
        expect(haveSameStructure(restored.stacks, before.canvas.document.stacks)).toBe(true);
        expect(getDocumentIndex(restored).nodes.map((entry) => entry.node)).toEqual(
          getDocumentIndex(before.canvas.document).nodes.map((entry) => entry.node)
        );
        expect(restored.selectedLayerId).toBe(before.canvas.document.selectedLayerId);
        const redone = applyCanvasProjectMutation(
          { ...before, canvas: { ...before.canvas, document: restored } },
          result.edit.forward
        ).canvas.document;
        expect(
          haveSameStructure(redone.stacks, after.canvas.document.stacks),
          `redo of ${JSON.stringify(command)}`
        ).toBe(true);
        project = after;
        applied += 1;
      }
      expect(applied).toBeGreaterThan(0);
    }
  );

  it('reached every outcome the generator is meant to produce', () => {
    expect(outcomes.locked).toBeGreaterThan(20);
    expect(outcomes.nestedDuplicate).toBeGreaterThan(5);
    expect(outcomes.nestedUngroup).toBeGreaterThan(5);
    expect(outcomes.reparentInside).toBeGreaterThan(50);
  });

  it('holds the depth boundary under random reparenting, grouping, inserting and duplicating', () => {
    const random = rng(99);
    created = 0;
    const chain = (depth: number): CanvasNodeContract =>
      depth === 0 ? layerContract(nextId(), 'raster') : groupContract(nextId(), [chain(depth - 1)]);
    let project = projectFor({
      ...createEmptyCanvasDocument(),
      selectedLayerId: null,
      stacks: stacksFrom([chain(CANVAS_MAX_NODE_DEPTH - 1), groupContract('spare', [layerContract('leaf', 'raster')])]),
    });
    const refusals = { duplicate: 0, group: 0, insert: 0, reparent: 0 };
    for (let step = 0; step < 120; step += 1) {
      const model = createDocumentModel(project.canvas.document, { editRevision: step, projectId: project.id });
      const groups = model.compileNodes().filter((node) => node.kind === 'group');
      const target = pick(random, groups);
      const roll = random();
      const command: DocumentCommand =
        roll < 0.25
          ? { beforeId: null, ids: ['spare'], parentId: target.id, type: 'reparent' }
          : roll < 0.5
            ? { groupId: nextId(), ids: [target.id], name: 'g', type: 'group' }
            : roll < 0.75
              ? {
                  aboveId: null,
                  insideId: target.id,
                  nodes: [groupContract(nextId(), [groupContract(nextId(), [layerContract(nextId(), 'raster')])])],
                  type: 'insert',
                }
              : { createId: nextId, ids: [target.id], type: 'duplicate' };
      const result = model.prepare(command);
      if (result.status === 'invalid-target' && result.reason === 'depth-exceeded') {
        refusals[command.type as keyof typeof refusals] += 1;
        continue;
      }
      if (result.status !== 'prepared') {
        continue;
      }
      project = applyCanvasProjectMutation(project, result.edit.forward);
      expect(project.canvas.document, `${command.type} was prepared but the reducer refused it`).not.toBe(
        model.document
      );
      assertInvariants(project.canvas.document);
    }
    expect(refusals.reparent).toBeGreaterThan(0);
    expect(refusals.group).toBeGreaterThan(0);
    expect(refusals.insert).toBeGreaterThan(0);
  });
});

describe('node-count boundary', () => {
  it('refuses every insert, duplicate and group that would pass 10,000 nodes and agrees with the reducer', () => {
    const random = rng(7);
    created = 0;
    const nodes = Array.from({ length: CANVAS_MAX_NODE_COUNT - 3 }, (_, index) =>
      layerContract(`b${index}`, index % 2 === 0 ? 'raster' : 'control')
    );
    let project = projectFor({ ...createEmptyCanvasDocument(), selectedLayerId: 'b0', stacks: stacksFrom(nodes) });
    const refusals = { duplicate: 0, group: 0, insert: 0 };
    let applied = 0;
    for (let step = 0; step < 40; step += 1) {
      const model = createDocumentModel(project.canvas.document, { editRevision: step, projectId: project.id });
      const roll = random();
      const count = 1 + Math.floor(random() * 5);
      const command: DocumentCommand =
        roll < 0.4
          ? {
              aboveId: null,
              nodes: Array.from({ length: count }, () => layerContract(nextId(), 'raster')),
              stack: 'raster',
              type: 'insert',
            }
          : roll < 0.8
            ? { createId: nextId, ids: Array.from({ length: count }, (_, i) => `b${i}`), type: 'duplicate' }
            : { groupId: nextId(), ids: ['b0', 'b2'], name: 'g', type: 'group' };
      const result = model.prepare(command);
      const size = getDocumentIndex(project.canvas.document).byId.size;
      const added =
        command.type === 'insert' ? command.nodes.length : command.type === 'duplicate' ? command.ids.length : 1;
      if (size + added > CANVAS_MAX_NODE_COUNT) {
        expect(result).toMatchObject({ reason: 'node-limit', status: 'invalid-target' });
        refusals[command.type as keyof typeof refusals] += 1;
        continue;
      }
      if (result.status !== 'prepared') {
        continue;
      }
      const next = applyCanvasProjectMutation(project, result.edit.forward);
      expect(next.canvas.document, `${command.type} was prepared but the reducer refused it`).not.toBe(
        project.canvas.document
      );
      expect(getDocumentIndex(next.canvas.document).byId.size).toBeLessThanOrEqual(CANVAS_MAX_NODE_COUNT);
      project = next;
      applied += 1;
    }
    expect(applied).toBeGreaterThan(0);
    expect(refusals.insert + refusals.duplicate + refusals.group).toBeGreaterThan(10);
    expect(getDocumentIndex(project.canvas.document).byId.size).toBe(CANVAS_MAX_NODE_COUNT);
  });
});
