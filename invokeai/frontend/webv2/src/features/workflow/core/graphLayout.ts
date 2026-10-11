/** Keep layout helpers independent of xyflow and UI contracts; callers adapt minimal node/edge shapes. */

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

export interface LayoutSizedNode extends LayoutGraphNode {
  height: number;
  width: number;
}

export interface SizedLayoutOptions {
  /** Horizontal gap between layers; defaults to 80. */
  layerGap?: number;
  /** Top-left corner of the arrangement; defaults to the origin. */
  origin?: { x: number; y: number };
  /** Vertical gap between the nodes of one layer; defaults to 40. */
  rowGap?: number;
}

const DEFAULT_LAYER_GAP = 80;
const DEFAULT_ROW_GAP = 40;

/**
 * Layered layout that respects each node's size: layers are columns as wide as their widest node, nodes stack top
 * to bottom within a layer, and a layer's rows follow the mean row of their parents in the layer before (ties keep
 * node-array order) so edges tend not to cross.
 */
export const getSizedLayeredPositions = (
  nodes: LayoutSizedNode[],
  edges: LayoutGraphEdge[],
  { layerGap = DEFAULT_LAYER_GAP, origin = { x: 0, y: 0 }, rowGap = DEFAULT_ROW_GAP }: SizedLayoutOptions = {}
): Record<string, { x: number; y: number }> => {
  const depths = getNodeDepths(nodes, edges);
  const layers = new Map<number, LayoutSizedNode[]>();

  for (const node of nodes) {
    const depth = depths.get(node.id) ?? 0;

    layers.set(depth, [...(layers.get(depth) ?? []), node]);
  }

  const parents = new Map<string, string[]>();

  for (const edge of edges) {
    parents.set(edge.targetNodeId, [...(parents.get(edge.targetNodeId) ?? []), edge.sourceNodeId]);
  }

  const positions: Record<string, { x: number; y: number }> = {};
  const rowOf = new Map<string, number>();
  let x = origin.x;

  for (const depth of [...layers.keys()].sort((a, b) => a - b)) {
    const members = layers.get(depth)!;
    const ordered = members
      .map((node, index) => {
        const parentRows = (parents.get(node.id) ?? [])
          .map((parent) => rowOf.get(parent))
          .filter((row): row is number => row !== undefined);
        const barycenter =
          parentRows.length === 0 ? index : parentRows.reduce((sum, row) => sum + row, 0) / parentRows.length;

        return { barycenter, index, node };
      })
      .sort((a, b) => a.barycenter - b.barycenter || a.index - b.index);
    let y = origin.y;
    let width = 0;

    ordered.forEach(({ node }, row) => {
      positions[node.id] = { x, y };
      rowOf.set(node.id, row);
      y += node.height + rowGap;
      width = Math.max(width, node.width);
    });

    x += width + layerGap;
  }

  return positions;
};
