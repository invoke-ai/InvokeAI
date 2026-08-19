/**
 * Generic node/edge layout helpers shared by the graph preview flow renderer
 * (`ui/graph-preview/GraphPreviewFlow.tsx`) and, eventually, Task 7's project
 * graph → preview converter. Deliberately xyflow-free and contract-free —
 * callers adapt their own node/edge shapes to the minimal shapes below.
 */

export interface LayoutGraphNode {
  id: string;
}

export interface LayoutGraphEdge {
  sourceNodeId: string;
  targetNodeId: string;
}

const LAYER_WIDTH = 300;
const ROW_HEIGHT = 100;

/** Longest-path-from-roots depth per node; cycle members settle at their first depth. */
export const getNodeDepths = (nodes: LayoutGraphNode[], edges: LayoutGraphEdge[]): Map<string, number> => {
  const depths = new Map<string, number>();
  const incoming = new Map<string, string[]>();

  for (const edge of edges) {
    incoming.set(edge.targetNodeId, [...(incoming.get(edge.targetNodeId) ?? []), edge.sourceNodeId]);
  }

  const resolve = (nodeId: string, seen: Set<string>): number => {
    const known = depths.get(nodeId);

    if (known !== undefined) {
      return known;
    }

    if (seen.has(nodeId)) {
      return 0;
    }

    seen.add(nodeId);

    const parents = incoming.get(nodeId) ?? [];
    const depth = parents.length === 0 ? 0 : Math.max(...parents.map((parent) => resolve(parent, seen))) + 1;

    depths.set(nodeId, depth);

    return depth;
  };

  for (const node of nodes) {
    resolve(node.id, new Set());
  }

  return depths;
};

/** Layered layout: x = depth * LAYER_WIDTH, y = row-within-depth * ROW_HEIGHT (row order follows node-array order). */
export const getLayeredPositions = (
  nodes: LayoutGraphNode[],
  edges: LayoutGraphEdge[]
): Record<string, { x: number; y: number }> => {
  const depths = getNodeDepths(nodes, edges);
  const rowsPerDepth = new Map<number, number>();
  const positions: Record<string, { x: number; y: number }> = {};

  for (const node of nodes) {
    const depth = depths.get(node.id) ?? 0;
    const row = rowsPerDepth.get(depth) ?? 0;

    rowsPerDepth.set(depth, row + 1);
    positions[node.id] = { x: depth * LAYER_WIDTH, y: row * ROW_HEIGHT };
  }

  return positions;
};

/** Node ids ordered by depth ascending; ties broken by input node-array order. */
export const getTopologicalOrder = (nodes: LayoutGraphNode[], edges: LayoutGraphEdge[]): string[] => {
  const depths = getNodeDepths(nodes, edges);

  return [...nodes]
    .map((node, index) => ({ depth: depths.get(node.id) ?? 0, id: node.id, index }))
    .sort((a, b) => a.depth - b.depth || a.index - b.index)
    .map((entry) => entry.id);
};
