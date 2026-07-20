export type GraphId = string;

export interface GraphNodeContract {
  id: string;
  type: string;
  inputs: Record<string, unknown>;
}

export interface GraphEdgeContract {
  id: string;
  sourceNodeId: string;
  sourceField: string;
  targetNodeId: string;
  targetField: string;
}

export interface GraphContract {
  id: GraphId;
  version: 1;
  label: string;
  nodes: GraphNodeContract[];
  edges: GraphEdgeContract[];
  updatedAt: string;
  backendGraph?: BackendGraphContract;
}

export interface BackendInvocationContract {
  id: string;
  type: string;
  [key: string]: unknown;
}

export interface BackendGraphEdgeContract {
  source: {
    node_id: string;
    field: string;
  };
  destination: {
    node_id: string;
    field: string;
  };
}

export interface BackendGraphContract {
  id: string;
  nodes: Record<string, BackendInvocationContract>;
  edges: BackendGraphEdgeContract[];
}
