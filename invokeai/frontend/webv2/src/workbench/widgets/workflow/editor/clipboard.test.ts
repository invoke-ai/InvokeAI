import type { ProjectGraphState, WorkflowEdge, WorkflowInvocationNode } from '@workbench/workflows/types';

import { createProjectGraph } from '@workbench/workflows/document';
import { describe, expect, it } from 'vitest';

import { buildDuplicateElements, buildPasteElements, copyNodesToClipboard, hasClipboardNodes } from './clipboard';

const createNode = (id: string, x: number, y: number): WorkflowInvocationNode => ({
  data: {
    inputs: { a: { label: '', name: 'a', value: 1 } },
    isIntermediate: true,
    isOpen: true,
    label: '',
    nodePack: 'invokeai',
    notes: '',
    type: 'add',
    useCache: true,
    version: '1.0.0',
  },
  id,
  position: { x, y },
  type: 'invocation',
});

const createEdge = (id: string, source: string, target: string): WorkflowEdge => ({
  id,
  source,
  sourceHandle: 'value',
  target,
  targetHandle: 'a',
  type: 'default',
});

const createDoc = (): ProjectGraphState => ({
  ...createProjectGraph('clipboard-test'),
  edges: [createEdge('edge-1', 'node-a', 'node-b'), createEdge('edge-2', 'node-outside', 'node-a')],
  nodes: [createNode('node-a', 0, 0), createNode('node-b', 100, 50), createNode('node-outside', -200, 0)],
});

describe('workflow clipboard', () => {
  it('copies selected nodes with only their internal edges', () => {
    const copiedCount = copyNodesToClipboard(createDoc(), ['node-a', 'node-b']);

    expect(copiedCount).toBe(2);
    expect(hasClipboardNodes()).toBe(true);

    const { edges, nodes } = buildPasteElements();

    expect(nodes).toHaveLength(2);
    // The edge from node-outside is not part of the copied set.
    expect(edges).toHaveLength(1);
  });

  it('pastes with fresh ids, remapped edges, and offset positions', () => {
    copyNodesToClipboard(createDoc(), ['node-a', 'node-b']);

    const { edges, nodes } = buildPasteElements();
    const pastedIds = new Set(nodes.map((node) => node.id));

    expect(pastedIds.has('node-a')).toBe(false);
    expect(pastedIds.has('node-b')).toBe(false);
    expect(edges[0]?.id).not.toBe('edge-1');
    expect(pastedIds.has(edges[0]?.source ?? '')).toBe(true);
    expect(pastedIds.has(edges[0]?.target ?? '')).toBe(true);
    expect(nodes[0]?.position).toEqual({ x: 32, y: 32 });

    // Pasting twice yields fresh ids each time.
    const second = buildPasteElements();

    expect(second.nodes.every((node) => !pastedIds.has(node.id))).toBe(true);
  });

  it('anchors the pasted group on an explicit position, preserving relative layout', () => {
    copyNodesToClipboard(createDoc(), ['node-a', 'node-b']);

    const { nodes } = buildPasteElements({ x: 500, y: 500 });

    expect(nodes.map((node) => node.position)).toEqual([
      { x: 500, y: 500 },
      { x: 600, y: 550 },
    ]);
  });

  it('duplicates in place without touching the clipboard', () => {
    copyNodesToClipboard(createDoc(), ['node-a']);

    const { edges, nodes } = buildDuplicateElements(createDoc(), ['node-a', 'node-b']);

    expect(nodes).toHaveLength(2);
    expect(edges).toHaveLength(1);
    // The clipboard still holds the single copied node.
    expect(buildPasteElements().nodes).toHaveLength(1);
  });

  it('copying nothing leaves the clipboard untouched', () => {
    copyNodesToClipboard(createDoc(), ['node-a']);

    expect(copyNodesToClipboard(createDoc(), ['missing-node'])).toBe(0);
    expect(buildPasteElements().nodes).toHaveLength(1);
  });
});
