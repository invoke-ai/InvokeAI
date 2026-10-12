/** Keep layout helpers independent of xyflow and UI contracts; callers adapt minimal node/edge shapes. */

export interface LayoutGraphNode {
  id: string;
}

export interface LayoutGraphEdge {
  sourceNodeId: string;
  targetNodeId: string;
}

/** Append to a map of lists, creating the list on first use. */
const pushTo = <K, V>(lists: Map<K, V[]>, key: K, value: V): void => {
  const list = lists.get(key);

  if (list) {
    list.push(value);
  } else {
    lists.set(key, [value]);
  }
};

const LAYER_WIDTH = 300;
const ROW_HEIGHT = 100;

/** Longest-path-from-roots depth per node; cycle members settle at their first depth. */
export const getNodeDepths = (nodes: LayoutGraphNode[], edges: LayoutGraphEdge[]): Map<string, number> => {
  const depths = new Map<string, number>();
  const incoming = new Map<string, string[]>();

  for (const edge of edges) {
    pushTo(incoming, edge.targetNodeId, edge.sourceNodeId);
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

/** Nodes grouped by undirected connectivity, each group in node-array order; groups ordered by first member. */
const getConnectedComponents = (nodes: LayoutSizedNode[], edges: LayoutGraphEdge[]): LayoutSizedNode[][] => {
  const neighbours = new Map<string, string[]>();

  for (const edge of edges) {
    pushTo(neighbours, edge.sourceNodeId, edge.targetNodeId);
    pushTo(neighbours, edge.targetNodeId, edge.sourceNodeId);
  }

  const componentOf = new Map<string, number>();
  const components: LayoutSizedNode[][] = [];

  for (const node of nodes) {
    if (componentOf.has(node.id)) {
      continue;
    }

    const index = components.length;
    const stack = [node.id];

    componentOf.set(node.id, index);

    while (stack.length > 0) {
      for (const next of neighbours.get(stack.pop()!) ?? []) {
        if (!componentOf.has(next)) {
          componentOf.set(next, index);
          stack.push(next);
        }
      }
    }

    components.push([]);
  }

  for (const node of nodes) {
    components[componentOf.get(node.id)!]!.push(node);
  }

  return components;
};

interface ComponentLayout {
  height: number;
  positions: Record<string, { x: number; y: number }>;
  width: number;
}

/**
 * Layered layout of one connected graph at the origin: layers are columns as wide as their widest node, nodes stack
 * top to bottom within a layer, and a layer's rows follow the mean row of their parents in the layer before (ties
 * keep node-array order) so edges tend not to cross.
 */
const layoutComponent = (
  nodes: LayoutSizedNode[],
  edges: LayoutGraphEdge[],
  layerGap: number,
  rowGap: number
): ComponentLayout => {
  const depths = getNodeDepths(nodes, edges);
  const layers = new Map<number, LayoutSizedNode[]>();

  for (const node of nodes) {
    const depth = depths.get(node.id) ?? 0;

    pushTo(layers, depth, node);
  }

  const parents = new Map<string, string[]>();

  for (const edge of edges) {
    pushTo(parents, edge.targetNodeId, edge.sourceNodeId);
  }

  const positions: Record<string, { x: number; y: number }> = {};
  const rowOf = new Map<string, number>();
  let x = 0;
  let height = 0;

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
    let y = 0;
    let width = 0;

    ordered.forEach(({ node }, row) => {
      positions[node.id] = { x, y };
      rowOf.set(node.id, row);
      height = Math.max(height, y + node.height);
      y += node.height + rowGap;
      width = Math.max(width, node.width);
    });

    x += width + layerGap;
  }

  return { height, positions, width: Math.max(0, x - layerGap) };
};

/**
 * Layered layout that respects each node's size. Each connected component is laid out on its own (see
 * `layoutComponent`); components then fill rows left to right in node-array order, wrapping at a width that keeps
 * the arrangement roughly square, so unconnected nodes form a grid rather than one tall column. A single connected
 * graph is simply its layered layout at the origin.
 */
export const getSizedLayeredPositions = (
  nodes: LayoutSizedNode[],
  edges: LayoutGraphEdge[],
  { layerGap = DEFAULT_LAYER_GAP, origin = { x: 0, y: 0 }, rowGap = DEFAULT_ROW_GAP }: SizedLayoutOptions = {}
): Record<string, { x: number; y: number }> => {
  const nodeIds = new Set(nodes.map((node) => node.id));
  const knownEdges = edges.filter((edge) => nodeIds.has(edge.sourceNodeId) && nodeIds.has(edge.targetNodeId));
  const components = getConnectedComponents(nodes, knownEdges).map((members) => {
    const memberIds = new Set(members.map((node) => node.id));

    return layoutComponent(
      members,
      knownEdges.filter((edge) => memberIds.has(edge.sourceNodeId)),
      layerGap,
      rowGap
    );
  });
  const gap = Math.max(layerGap, rowGap);
  const perRow = Math.ceil(Math.sqrt(components.length));
  const meanWidth = components.reduce((sum, component) => sum + component.width, 0) / components.length;
  const rowBudget = Math.max(...components.map((component) => component.width), perRow * (meanWidth + gap) - gap);
  const positions: Record<string, { x: number; y: number }> = {};
  let x = 0;
  let y = 0;
  let rowHeight = 0;

  for (const component of components) {
    if (x > 0 && x + component.width > rowBudget) {
      x = 0;
      y += rowHeight + gap;
      rowHeight = 0;
    }

    for (const [id, position] of Object.entries(component.positions)) {
      positions[id] = { x: origin.x + x + position.x, y: origin.y + y + position.y };
    }

    x += component.width + gap;
    rowHeight = Math.max(rowHeight, component.height);
  }

  return positions;
};

export interface ArrangeNode extends LayoutGraphNode {
  position: { x: number; y: number };
}

export interface ArrangeEdge {
  source: string;
  target: string;
  type: string;
}

export interface ArrangedLayout {
  /** The arranged nodes' footprint, for the view to fit before the rendered positions catch up. */
  bounds: { height: number; width: number; x: number; y: number };
  positions: Record<string, { x: number; y: number }>;
}

/** The size assumed for a node the view has not measured yet (a large graph renders only what is visible). */
export const UNMEASURED_ARRANGE_SIZE = { height: 160, width: 288 } as const;

/**
 * The editor's Arrange command: lays out `selectedIds` (every node when null) with `getSizedLayeredPositions`,
 * anchored at the arranged nodes' current top-left corner so the result stays where the author was working. Nodes
 * outside the selection keep their positions, and edges to them do not rank the arranged ones. Loop linkages are
 * ignored too: they point backwards, and ranking by them would fold a loop body onto its own head.
 */
export const getArrangedLayout = (
  nodes: readonly ArrangeNode[],
  edges: readonly ArrangeEdge[],
  selectedIds: ReadonlySet<string> | null,
  measuredSizes: ReadonlyMap<string, { height?: number; width?: number } | undefined>
): ArrangedLayout => {
  const targets = nodes.filter((node) => !selectedIds || selectedIds.has(node.id));
  const origin = targets.reduce(
    (corner, node) => ({ x: Math.min(corner.x, node.position.x), y: Math.min(corner.y, node.position.y) }),
    { x: Infinity, y: Infinity }
  );
  const sized = targets.map((node) => {
    const measured = measuredSizes.get(node.id);

    return {
      height: measured?.height || UNMEASURED_ARRANGE_SIZE.height,
      id: node.id,
      width: measured?.width || UNMEASURED_ARRANGE_SIZE.width,
    };
  });
  const layoutEdges = edges
    .filter((edge) => edge.type !== 'loop_linkage')
    .map((edge) => ({ sourceNodeId: edge.source, targetNodeId: edge.target }));
  const positions = getSizedLayeredPositions(sized, layoutEdges, { origin });
  const rect = sized.reduce(
    (current, node) => {
      const position = positions[node.id]!;

      return {
        maxX: Math.max(current.maxX, position.x + node.width),
        maxY: Math.max(current.maxY, position.y + node.height),
        minX: Math.min(current.minX, position.x),
        minY: Math.min(current.minY, position.y),
      };
    },
    { maxX: -Infinity, maxY: -Infinity, minX: Infinity, minY: Infinity }
  );

  return {
    bounds: { height: rect.maxY - rect.minY, width: rect.maxX - rect.minX, x: rect.minX, y: rect.minY },
    positions,
  };
};
