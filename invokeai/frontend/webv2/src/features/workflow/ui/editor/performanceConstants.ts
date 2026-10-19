export const WORKFLOW_LARGE_GRAPH_NODE_COUNT = 300;
export const WORKFLOW_LARGE_GRAPH_EDGE_COUNT = 500;

export const WORKFLOW_MINIMAP_DELAY_MS = 450;
export const WORKFLOW_INITIAL_RENDER_NODE_COUNT = 120;

export const isLargeWorkflowGraph = ({ edgeCount, nodeCount }: { edgeCount: number; nodeCount: number }): boolean =>
  nodeCount >= WORKFLOW_LARGE_GRAPH_NODE_COUNT || edgeCount >= WORKFLOW_LARGE_GRAPH_EDGE_COUNT;
