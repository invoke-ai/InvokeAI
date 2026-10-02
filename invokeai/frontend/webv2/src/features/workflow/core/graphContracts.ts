export interface WorkflowBackendInvocation {
  id: string;
  type: string;
  [key: string]: unknown;
}

export interface WorkflowBackendGraph {
  id: string;
  nodes: Record<string, WorkflowBackendInvocation>;
  edges: Array<{
    type: 'default' | 'loop_linkage';
    source: { node_id: string; field: string };
    destination: { node_id: string; field: string };
  }>;
}

export interface CompiledWorkflowGraph {
  id: string;
  version: 1;
  label: string;
  nodes: Array<{ id: string; type: string; inputs: Record<string, unknown> }>;
  edges: Array<{
    id: string;
    type: 'default' | 'loop_linkage';
    sourceNodeId: string;
    sourceField: string;
    targetNodeId: string;
    targetField: string;
  }>;
  updatedAt: string;
  backendGraph: WorkflowBackendGraph;
}
