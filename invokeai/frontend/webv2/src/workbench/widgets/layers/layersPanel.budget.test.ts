import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';

import { compileDocumentNodes, getDocumentIndex, isGroupNode } from '@workbench/canvas-engine/api';
import {
  createLargeFlatDocument,
  createLargeTreeDocument,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import {
  getDocumentModelDiagnostics,
  resetDocumentModelDiagnostics,
} from '@workbench/canvas-engine/document-model/documentModel';
import { describe, expect, it } from 'vitest';

import { flattenPanelRows, navigateTree } from './layerPanelRows';
import { outermostRowIds, projectLayerDrop } from './layerTreeDrop';
import { buildLayerStackRows } from './layerTreeRows';

const NODE_COUNT = 2_000;

type Annotate = (message: string) => unknown;

const timed = <T>(annotate: Annotate, label: string, run: () => T): T => {
  const start = performance.now();
  const value = run();
  annotate(`${label}: ${(performance.now() - start).toFixed(2)}ms`);
  return value;
};

const groupIds = (document: CanvasDocumentContractV3): string[] =>
  getDocumentIndex(document)
    .nodes.filter((entry) => isGroupNode(entry.node))
    .map((entry) => entry.node.id);

describe(`layers panel budgets over ${NODE_COUNT} nodes`, () => {
  it('renders exactly the roots when collapsed and every node when expanded, in one visit each', ({ annotate }) => {
    const document = createLargeTreeDocument(NODE_COUNT);
    const index = getDocumentIndex(document);
    const roots = index.nodes.filter((entry) => entry.parentId === null).length;
    resetDocumentModelDiagnostics();
    const collapsed = timed(annotate, 'collapsed rows', () => buildLayerStackRows(document.stacks, new Set()));
    expect(Object.values(collapsed).reduce((total, stack) => total + stack.rows.length, 0)).toBe(roots);
    expect(getDocumentModelDiagnostics().nodesCompiled).toBe(NODE_COUNT);

    const expanded = timed(annotate, 'expanded rows', () =>
      buildLayerStackRows(document.stacks, new Set(groupIds(document)))
    );
    expect(Object.values(expanded).reduce((total, stack) => total + stack.rows.length, 0)).toBe(NODE_COUNT);
    expect(getDocumentModelDiagnostics().nodesCompiled).toBe(NODE_COUNT);
    const flat = flattenPanelRows(expanded, [], () => false);
    expect(flat.filter((row) => row.kind === 'header')).toHaveLength(4);
    expect(flat.filter((row) => row.kind === 'node')).toHaveLength(NODE_COUNT);
  });

  it('keeps every row identity across a selection change and all but the affected rows across a local expand', () => {
    const document = createLargeTreeDocument(NODE_COUNT);
    const groups = groupIds(document);
    const expanded = new Set(groups);
    const before = buildLayerStackRows(document.stacks, expanded);
    // Selection lives beside the document, so nothing here can change.
    const again = buildLayerStackRows(document.stacks, expanded);
    for (const stack of Object.keys(before) as (keyof typeof before)[]) {
      before[stack].rows.forEach((row, position) => expect(again[stack].rows[position]).toBe(row));
    }
    const target = groups.find((id) => getDocumentIndex(document).byId.get(id)!.path.length === 1)!;
    const collapsedOne = new Set(groups);
    collapsedOne.delete(target);
    const after = buildLayerStackRows(document.stacks, collapsedOne);
    const descendants = new Set(
      getDocumentIndex(document)
        .nodes.filter((entry) => entry.path.includes(target))
        .map((entry) => entry.node.id)
    );
    const beforeById = new Map(
      Object.values(before).flatMap((stack) => stack.rows.map((row) => [row.id, row] as const))
    );
    let changed = 0;
    for (const stack of Object.values(after)) {
      for (const row of stack.rows) {
        expect(descendants.has(row.id)).toBe(false);
        if (beforeById.get(row.id) !== row) {
          changed += 1;
          expect(row.id).toBe(target);
        }
      }
    }
    expect(changed).toBe(1);
  });

  it('projects a large multi-selection drag in linear time and keeps semantic order', ({ annotate }) => {
    const document = createLargeFlatDocument(NODE_COUNT * 2, null);
    const rows = buildLayerStackRows(document.stacks, new Set()).raster.rows;
    const selected = new Set(rows.filter((_, position) => position % 2 === 0).map((row) => row.id));
    const activeIds = timed(annotate, 'outermost rows', () => outermostRowIds(rows, selected));
    expect(activeIds).toHaveLength(selected.size);
    const target = timed(annotate, 'drop projection', () =>
      projectLayerDrop({ activeIds, depthOffset: 0, edge: 'below', overId: rows[rows.length - 1]!.id, rows })
    );
    expect(target).toMatchObject({ beforeId: null, depth: 0, parentId: null });
    expect(target!.ids).toEqual(activeIds);
  });

  it('navigates the flat list in bounded steps', () => {
    const document = createLargeTreeDocument(NODE_COUNT);
    const rows = flattenPanelRows(buildLayerStackRows(document.stacks, new Set(groupIds(document))), [], () => false);
    const first = rows.find((row) => row.kind === 'node')!;
    expect(navigateTree(rows, first.key, 'End')).toEqual({ focus: rows[rows.length - 1]!.key });
    expect(navigateTree(rows, rows[rows.length - 1]!.key, 'Home')).toEqual({ focus: rows[0]!.key });
    expect(navigateTree(rows, rows[0]!.key, 'ArrowRight')).toEqual({ focus: first.key });
    const nodes = compileDocumentNodes(document);
    const nested = nodes.find((node) => node.depth >= 2 && node.kind === 'leaf')!;
    expect(navigateTree(rows, nested.id, 'ArrowLeft')).toEqual({ focus: nested.parentId });
  });
});
