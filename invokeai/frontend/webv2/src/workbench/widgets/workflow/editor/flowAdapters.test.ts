import type { ProjectGraphState, WorkflowEdge, WorkflowInvocationNode } from '@workbench/workflows/types';

import { buildCurrentImageNode, createProjectGraph } from '@workbench/workflows/document';
import { describe, expect, it } from 'vitest';

import { toFlowEdges, toFlowNodes, withNodeSelection } from './flowAdapters';

const createNode = (id: string): WorkflowInvocationNode => ({
  data: {
    inputs: {},
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
  position: { x: 0, y: 0 },
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

const createDoc = (overrides?: Partial<ProjectGraphState>): ProjectGraphState => ({
  ...createProjectGraph('adapters-test'),
  edges: [createEdge('e1', 'a', 'b')],
  nodes: [createNode('a'), createNode('b')],
  ...overrides,
});

describe('flowAdapters identity preservation', () => {
  it('reuses unchanged node objects across rebuilds (memo-friendly)', () => {
    const doc = createDoc();
    const first = toFlowNodes(doc);
    const second = toFlowNodes(doc, first);

    expect(second[0]).toBe(first[0]);
    expect(second[1]).toBe(first[1]);
  });

  it('replaces only the node whose document node changed', () => {
    const doc = createDoc();
    const first = toFlowNodes(doc);
    const movedNodeB = { ...doc.nodes[1]!, position: { x: 50, y: 50 } };
    const second = toFlowNodes({ ...doc, nodes: [doc.nodes[0]!, movedNodeB] }, first);

    expect(second[0]).toBe(first[0]);
    expect(second[1]).not.toBe(first[1]);
  });

  it('recomputes a node when its incoming connections change', () => {
    const doc = createDoc();
    const first = toFlowNodes(doc);
    // Drop the edge into node b; its connectedTargetHandles must update.
    const second = toFlowNodes({ ...doc, edges: [] }, first);
    const nodeB = second[1];

    expect(nodeB?.type === 'invocation' && nodeB.data.connectedTargetHandles).toEqual([]);
    expect(second[1]).not.toBe(first[1]);
    // Node a had no incoming edges either way — still identity-stable.
    expect(second[0]).toBe(first[0]);
  });

  it('precomputes exposed field names onto invocation node data', () => {
    const doc = createDoc();
    const exposed = {
      ...doc,
      form: {
        ...doc.form,
        elements: {
          ...doc.form.elements,
          'field-1': {
            data: { fieldIdentifier: { fieldName: 'a', nodeId: 'b' }, showDescription: false },
            id: 'field-1',
            parentId: doc.form.rootElementId,
            type: 'node-field' as const,
          },
        },
      },
    };
    const nodes = toFlowNodes(exposed);
    const nodeB = nodes[1];

    expect(nodeB?.type === 'invocation' && nodeB.data.exposedFieldNames).toEqual(['a']);
  });

  it('carries selection across rebuilds and toggles it without copying unchanged nodes', () => {
    const doc = createDoc();
    const selected = withNodeSelection(toFlowNodes(doc), new Set(['a']));

    expect(selected[0]?.selected).toBe(true);

    const rebuilt = toFlowNodes(doc, selected);

    expect(rebuilt[0]?.selected).toBe(true);

    // Re-applying the same selection set returns identical node objects.
    const reselected = withNodeSelection(selected, new Set(['a']));

    expect(reselected[0]).toBe(selected[0]);
  });

  it('builds current_image nodes and reuses them when unchanged', () => {
    const currentImage = buildCurrentImageNode({ x: 0, y: 0 });
    const doc = createDoc({ nodes: [createNode('a'), currentImage] });
    const first = toFlowNodes(doc);

    expect(first[1]?.type).toBe('current_image');
    expect(toFlowNodes(doc, first)[1]).toBe(first[1]);
  });

  it('reuses unchanged edge objects but replaces edges on type change', () => {
    const doc = createDoc();
    const first = toFlowEdges(doc, [], 'default');
    const same = toFlowEdges(doc, first, 'default');

    expect(same[0]).toBe(first[0]);

    const restyled = toFlowEdges(doc, first, 'straight');

    expect(restyled[0]).not.toBe(first[0]);
    expect(restyled[0]?.type).toBe('straight');
  });
});
