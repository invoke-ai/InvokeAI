import type { CanvasDocumentContractV3, CanvasNodeContract, PreparedDocumentEdit } from '@workbench/canvas-engine/api';
import type { Project } from '@workbench/projectContracts';
import type { CanvasPreparedEngine } from '@workbench/widgets/canvas/useStructuralCommit';

import { createDocumentModel, getDocumentIndex } from '@workbench/canvas-engine/api';
import {
  groupContract as group,
  layerContract as layer,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { describe, expect, it } from 'vitest';

import { canRemoveFromGroup, removeFromGroup } from './layerGroupCommands';

const projectWith = (nodes: CanvasNodeContract[]): Project =>
  applyCanvasProjectMutation(createInitialWorkbenchState().projects[0]!, {
    document: { ...createEmptyCanvasDocument(), selectedLayerId: null, stacks: stacksFrom(nodes) },
    type: 'replaceCanvasDocument',
  });

/** Stacks top first, each node in preorder, indented by depth. */
const outline = (document: CanvasDocumentContractV3): string[] =>
  getDocumentIndex(document).nodes.map((entry) => `${'  '.repeat(entry.path.length)}${entry.node.id}`);

/** A prepared-edit engine over a real document model whose commits apply through the real reducer. */
const createEngine = (initial: Project) => {
  let project = initial;
  const edits: PreparedDocumentEdit[] = [];
  const engine: CanvasPreparedEngine = {
    document: { model: () => createDocumentModel(project.canvas.document, { editRevision: 0, projectId: project.id }) },
    layers: {
      commitPrepared: (_label, edit) => {
        edits.push(edit);
        project = applyCanvasProjectMutation(project, edit.forward);
        return { status: 'committed' };
      },
      endStructuralPreview: () => undefined,
    },
  };
  return {
    document: () => project.canvas.document,
    edits,
    engine,
    undo: () => {
      project = applyCanvasProjectMutation(project, edits.at(-1)!.inverse);
    },
  };
};

/** raster: r1, G[r2, H[r3, r4], r5], r6 */
const tree = (overrides: { lockH?: boolean } = {}): CanvasNodeContract[] => [
  layer('r1'),
  group('G', [
    layer('r2'),
    group('H', [layer('r3'), layer('r4')], { isLocked: overrides.lockH ?? false }),
    layer('r5'),
  ]),
  layer('r6'),
];

describe('removeFromGroup', () => {
  it('moves a leaf directly above its group in the parent, and one undo puts it back', () => {
    const harness = createEngine(projectWith(tree()));
    const before = outline(harness.document());

    expect(removeFromGroup(harness.engine, ['r5'], 'Remove from group')).toEqual({ status: 'committed' });

    expect(outline(harness.document())).toEqual(['r1', 'r5', 'G', '  r2', '  H', '    r3', '    r4', 'r6']);
    expect(harness.edits).toHaveLength(1);
    harness.undo();
    expect(outline(harness.document())).toEqual(before);
  });

  it('lifts several leaves of a nested group above it in document order, whatever the selection order', () => {
    const harness = createEngine(projectWith(tree()));

    removeFromGroup(harness.engine, ['r4', 'r3'], 'Remove from group');

    expect(outline(harness.document())).toEqual(['r1', 'G', '  r2', '  r3', '  r4', '  H', '  r5', 'r6']);
  });

  it('refuses leaves inside a locked group and changes nothing', () => {
    const harness = createEngine(projectWith(tree({ lockH: true })));
    const before = outline(harness.document());

    expect(canRemoveFromGroup(harness.engine.document.model(), ['r3'])).toBe(false);
    expect(removeFromGroup(harness.engine, ['r3'], 'Remove from group')).toMatchObject({
      refusal: { status: 'locked' },
      status: 'refused',
    });
    expect(outline(harness.document())).toEqual(before);
  });

  it.each([
    ['a top-level leaf', ['r1']],
    ['leaves of different groups', ['r2', 'r3']],
    ['a group', ['H']],
    ['an absent id', ['r5', 'ghost']],
  ])('is unavailable for %s', (_name, ids) => {
    const harness = createEngine(projectWith(tree()));

    expect(canRemoveFromGroup(harness.engine.document.model(), ids)).toBe(false);
    expect(removeFromGroup(harness.engine, ids, 'Remove from group')).toEqual({ status: 'unchanged' });
    expect(harness.edits).toHaveLength(0);
  });

  it('is available for leaves of one unlocked group', () => {
    const harness = createEngine(projectWith(tree()));

    expect(canRemoveFromGroup(harness.engine.document.model(), ['r2', 'r5'])).toBe(true);
  });
});
