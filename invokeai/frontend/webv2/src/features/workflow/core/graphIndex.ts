import type { WorkflowEdge, WorkflowNode } from './types';

import { CONNECTOR_INPUT_HANDLE, CONNECTOR_OUTPUT_HANDLE } from './connectorHandles';

export interface WorkflowGraphIndex {
  connectorInputById: Map<string, WorkflowEdge>;
  connectorOutputsById: Map<string, WorkflowEdge[]>;
  edgesBySource: Map<string, WorkflowEdge[]>;
  edgesByTarget: Map<string, WorkflowEdge[]>;
  nodesById: Map<string, WorkflowNode>;
}

const append = <K, V>(map: Map<K, V[]>, key: K, value: V): void => {
  const values = map.get(key);

  if (values) {
    values.push(value);
    return;
  }

  map.set(key, [value]);
};

export const createWorkflowGraphIndex = (nodes: WorkflowNode[], edges: WorkflowEdge[]): WorkflowGraphIndex => {
  const index: WorkflowGraphIndex = {
    connectorInputById: new Map(),
    connectorOutputsById: new Map(),
    edgesBySource: new Map(),
    edgesByTarget: new Map(),
    nodesById: new Map(),
  };

  for (const node of nodes) {
    index.nodesById.set(node.id, node);
  }

  for (const edge of edges) {
    append(index.edgesBySource, edge.source, edge);
    append(index.edgesByTarget, edge.target, edge);

    if (edge.type !== 'default') {
      continue;
    }

    if (edge.targetHandle === CONNECTOR_INPUT_HANDLE) {
      if (!index.connectorInputById.has(edge.target)) {
        index.connectorInputById.set(edge.target, edge);
      }
    }

    if (edge.sourceHandle === CONNECTOR_OUTPUT_HANDLE) {
      append(index.connectorOutputsById, edge.source, edge);
    }
  }

  return index;
};
