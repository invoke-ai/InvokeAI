import type {
  ProjectGraphState,
  WorkflowConnectorNode,
  WorkflowEdge,
  WorkflowInvocationNode,
} from '@features/workflow/contracts';

import { createProjectGraph } from '@features/workflow/utility';
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

const createConnector = (id: string): WorkflowConnectorNode => ({
  data: { label: '' },
  id,
  position: { x: 50, y: 0 },
  type: 'connector',
});

const connectorEdge = (id: string, source: string, target: string): WorkflowEdge => ({
  id,
  source,
  sourceHandle: source.startsWith('connector') ? 'out' : 'value',
  target,
  targetHandle: target.startsWith('connector') ? 'in' : 'a',
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

    const { nodes } = buildPasteElements({ placement: { align: 'top-left', point: { x: 500, y: 500 } } });

    expect(nodes.map((node) => node.position)).toEqual([
      { x: 500, y: 500 },
      { x: 600, y: 550 },
    ]);
  });

  it("centers the group's node positions on a point, preserving relative layout", () => {
    copyNodesToClipboard(createDoc(), ['node-a', 'node-b']);

    const { nodes } = buildPasteElements({ placement: { align: 'center', point: { x: 500, y: 500 } } });

    // Positions span (0,0)–(100,50); their middle (50,25) lands on the point.
    expect(nodes.map((node) => node.position)).toEqual([
      { x: 450, y: 475 },
      { x: 550, y: 525 },
    ]);
  });

  it('keeps edges from outside nodes only when pasting with connections', () => {
    copyNodesToClipboard(createDoc(), ['node-a']);

    expect(buildPasteElements().edges).toEqual([]);

    const { edges, nodes } = buildPasteElements({ withConnections: true });

    expect(edges).toHaveLength(1);
    expect(edges[0]).toMatchObject({ source: 'node-outside', target: nodes[0]?.id });
    expect(edges[0]?.id).not.toBe('edge-2');
  });

  it('never carries a loop linkage into a pasted copy', () => {
    copyNodesToClipboard(
      {
        ...createDoc(),
        edges: [
          { ...createEdge('linkage', 'node-outside', 'node-a'), targetHandle: 'loop_linkage', type: 'loop_linkage' },
          { ...createEdge('aliased-linkage', 'node-outside', 'node-a'), targetHandle: 'loop_linkage' },
        ],
      },
      ['node-a']
    );

    expect(buildPasteElements({ withConnections: true }).edges).toEqual([]);
  });

  it('copies an unselected connector that bridges two copied nodes', () => {
    const doc: ProjectGraphState = {
      ...createProjectGraph('clipboard-connector-test'),
      edges: [
        connectorEdge('a-to-c1', 'node-a', 'connector-1'),
        connectorEdge('c1-to-c2', 'connector-1', 'connector-2'),
        connectorEdge('c2-to-b', 'connector-2', 'node-b'),
        connectorEdge('outside-to-c3', 'node-outside', 'connector-3'),
        connectorEdge('c3-to-b', 'connector-3', 'node-b'),
        connectorEdge('a-to-c4', 'node-a', 'connector-4'),
        connectorEdge('c4-to-outside', 'connector-4', 'node-outside'),
      ],
      nodes: [
        createNode('node-a', 0, 0),
        createNode('node-b', 100, 50),
        createNode('node-outside', -200, 0),
        createConnector('connector-1'),
        createConnector('connector-2'),
        createConnector('connector-3'),
        createConnector('connector-4'),
      ],
    };

    expect(copyNodesToClipboard(doc, ['node-a', 'node-b'])).toBe(2);

    const { edges, nodes } = buildPasteElements();
    const originalIdByPastedId = new Map(
      nodes.map((node, index) => [node.id, ['node-a', 'node-b', 'connector-1', 'connector-2'][index]])
    );

    // Connectors 3 (fed from outside) and 4 (leading outside) do not bridge the copy.
    expect(nodes.map((node) => node.type)).toEqual(['invocation', 'invocation', 'connector', 'connector']);
    expect(
      edges.map((edge) => `${originalIdByPastedId.get(edge.source)}->${originalIdByPastedId.get(edge.target)}`)
    ).toEqual(['node-a->connector-1', 'connector-1->connector-2', 'connector-2->node-b']);
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
