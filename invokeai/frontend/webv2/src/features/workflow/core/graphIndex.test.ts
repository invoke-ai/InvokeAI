import { describe, expect, it } from 'vitest';

import type { WorkflowConnectorNode, WorkflowEdge, WorkflowInvocationNode } from './types';

import { createWorkflowGraphIndex } from './graphIndex';

const makeNode = (id: string): WorkflowInvocationNode => ({
  data: {
    inputs: {},
    isIntermediate: true,
    isOpen: true,
    label: '',
    nodePack: 'invokeai',
    notes: '',
    type: 'number',
    useCache: true,
    version: '1.0.0',
  },
  id,
  position: { x: 0, y: 0 },
  type: 'invocation',
});

const makeConnector = (id: string): WorkflowConnectorNode => ({
  data: { label: '' },
  id,
  position: { x: 0, y: 0 },
  type: 'connector',
});

const edge = (
  id: string,
  source: string,
  sourceHandle: string,
  target: string,
  targetHandle: string
): WorkflowEdge => ({
  id,
  source,
  sourceHandle,
  target,
  targetHandle,
  type: 'default',
});

describe('createWorkflowGraphIndex', () => {
  it('indexes nodes and connector edges for repeated lookup', () => {
    const source = makeNode('source');
    const target = makeNode('target');
    const connector = makeConnector('connector');
    const connectorInput = edge('connector-input', 'source', 'value', 'connector', 'in');
    const connectorOutput = edge('connector-output', 'connector', 'out', 'target', 'value');
    const index = createWorkflowGraphIndex([source, connector, target], [connectorInput, connectorOutput]);

    expect(index.nodesById.get('source')).toBe(source);
    expect(index.edgesBySource.get('connector')).toEqual([connectorOutput]);
    expect(index.edgesByTarget.get('connector')).toEqual([connectorInput]);
    expect(index.connectorInputById.get('connector')).toBe(connectorInput);
    expect(index.connectorOutputsById.get('connector')).toEqual([connectorOutput]);
  });

  it('preserves first-match connector input semantics for invalid duplicate inputs', () => {
    const sourceA = makeNode('source-a');
    const sourceB = makeNode('source-b');
    const connector = makeConnector('connector');
    const firstInput = edge('first-input', 'source-a', 'value', 'connector', 'in');
    const secondInput = edge('second-input', 'source-b', 'value', 'connector', 'in');
    const index = createWorkflowGraphIndex([sourceA, sourceB, connector], [firstInput, secondInput]);

    expect(index.connectorInputById.get('connector')).toBe(firstInput);
  });
});
