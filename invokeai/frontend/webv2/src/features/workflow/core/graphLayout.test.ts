import { describe, expect, it } from 'vitest';

import type { LayoutGraphEdge, LayoutGraphNode } from './graphLayout';

import { getLayeredPositions, getNodeDepths, getTopologicalOrder } from './graphLayout';

// Diamond: a -> b, a -> c, b -> d, c -> d
const DIAMOND_NODES: LayoutGraphNode[] = [{ id: 'a' }, { id: 'b' }, { id: 'c' }, { id: 'd' }];
const DIAMOND_EDGES: LayoutGraphEdge[] = [
  { sourceNodeId: 'a', targetNodeId: 'b' },
  { sourceNodeId: 'a', targetNodeId: 'c' },
  { sourceNodeId: 'b', targetNodeId: 'd' },
  { sourceNodeId: 'c', targetNodeId: 'd' },
];

describe('getNodeDepths', () => {
  it('assigns longest-path-from-roots depth for a diamond graph', () => {
    const depths = getNodeDepths(DIAMOND_NODES, DIAMOND_EDGES);

    expect(Object.fromEntries(depths)).toEqual({ a: 0, b: 1, c: 1, d: 2 });
  });

  it('terminates on a cycle instead of recursing forever, breaking the loop at re-entry', () => {
    const nodes: LayoutGraphNode[] = [{ id: 'x' }, { id: 'y' }];
    const edges: LayoutGraphEdge[] = [
      { sourceNodeId: 'x', targetNodeId: 'y' },
      { sourceNodeId: 'y', targetNodeId: 'x' },
    ];

    const depths = getNodeDepths(nodes, edges);

    expect(depths.size).toBe(2);
    expect(Number.isFinite(depths.get('x'))).toBe(true);
    expect(Number.isFinite(depths.get('y'))).toBe(true);
  });
});

describe('getTopologicalOrder', () => {
  it('orders by depth ascending, then by input node-array order within a depth', () => {
    expect(getTopologicalOrder(DIAMOND_NODES, DIAMOND_EDGES)).toEqual(['a', 'b', 'c', 'd']);
  });

  it('preserves array order for nodes at the same depth even when reversed', () => {
    const nodes: LayoutGraphNode[] = [{ id: 'c' }, { id: 'b' }, { id: 'a' }];
    const edges: LayoutGraphEdge[] = [
      { sourceNodeId: 'a', targetNodeId: 'b' },
      { sourceNodeId: 'a', targetNodeId: 'c' },
    ];

    // a is depth 0; b and c are both depth 1 — order should follow the input array: c, b.
    expect(getTopologicalOrder(nodes, edges)).toEqual(['a', 'c', 'b']);
  });
});

describe('getLayeredPositions', () => {
  it('places nodes at x = depth * 300, rows spaced by 100 within a depth', () => {
    const positions = getLayeredPositions(DIAMOND_NODES, DIAMOND_EDGES);

    expect(positions.a).toEqual({ x: 0, y: 0 });
    expect(positions.b).toEqual({ x: 300, y: 0 });
    expect(positions.c).toEqual({ x: 300, y: 100 });
    expect(positions.d).toEqual({ x: 600, y: 0 });
  });
});
