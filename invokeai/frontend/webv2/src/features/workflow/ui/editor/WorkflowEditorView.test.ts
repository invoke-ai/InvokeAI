import { describe, expect, it } from 'vitest';

import type { WorkflowFlowEdge, WorkflowFlowNode } from './flowAdapters';

import { WORKFLOW_INITIAL_RENDER_NODE_COUNT } from './performanceConstants';
import { getInitialRenderFlowModel } from './WorkflowEditorView';

const createNode = (id: string, x: number): WorkflowFlowNode => ({
  data: { documentNode: { data: { label: '', notes: '' }, id, position: { x, y: 0 }, type: 'notes' } },
  id,
  position: { x, y: 0 },
  type: 'notes',
});

const createEdge = (id: string, source: string, target: string): WorkflowFlowEdge => ({
  data: {
    fieldTypeLabel: 'Unknown',
    pathType: 'default',
    stroke: 'var(--xy-edge-stroke)',
    strokeWidth: 2,
    tooltip: 'Unknown field type',
  },
  id,
  source,
  sourceHandle: 'value',
  target,
  targetHandle: 'value',
  type: 'default',
});

describe('getInitialRenderFlowModel', () => {
  it('keeps the nearest initial nodes and filters edges to that window', () => {
    const nodes = Array.from({ length: WORKFLOW_INITIAL_RENDER_NODE_COUNT + 2 }, (_, index) =>
      createNode(`node-${index}`, index)
    );
    const model = {
      nodes,
      edges: [
        createEdge('inside', 'node-0', 'node-1'),
        createEdge('outside', 'node-0', `node-${WORKFLOW_INITIAL_RENDER_NODE_COUNT + 1}`),
      ],
    };

    const windowed = getInitialRenderFlowModel(model, { x: 0, y: 0, zoom: 1 });

    expect(windowed.nodes).toHaveLength(WORKFLOW_INITIAL_RENDER_NODE_COUNT);
    expect(windowed.nodes.map((node) => node.id)).toContain('node-0');
    expect(windowed.nodes.map((node) => node.id)).not.toContain(`node-${WORKFLOW_INITIAL_RENDER_NODE_COUNT + 1}`);
    expect(windowed.edges.map((edge) => edge.id)).toEqual(['inside']);
  });
});
