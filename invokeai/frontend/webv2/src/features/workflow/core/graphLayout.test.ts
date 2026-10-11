import { describe, expect, it } from 'vitest';

import type { LayoutGraphEdge, LayoutGraphNode } from './graphLayout';

import { getLayeredPositions, getNodeDepths, getSizedLayeredPositions, getTopologicalOrder } from './graphLayout';

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

describe('getSizedLayeredPositions', () => {
  it('stacks each layer by node height and places the next layer past the widest node', () => {
    const nodes = [
      { height: 100, id: 'a', width: 200 },
      { height: 300, id: 'b', width: 250 },
      { height: 120, id: 'c', width: 180 },
      { height: 80, id: 'd', width: 200 },
    ];

    expect(
      getSizedLayeredPositions(nodes, DIAMOND_EDGES, { layerGap: 50, origin: { x: 10, y: 20 }, rowGap: 10 })
    ).toEqual({
      a: { x: 10, y: 20 },
      b: { x: 260, y: 20 },
      c: { x: 260, y: 330 },
      d: { x: 560, y: 20 },
    });
  });

  it('orders a layer by the mean row of its parents so edges do not cross', () => {
    // p -> y and q -> x, listed in the "wrong" order: x and y swap rows to follow their parents.
    const nodes = [
      { height: 10, id: 'p', width: 10 },
      { height: 10, id: 'q', width: 10 },
      { height: 10, id: 'x', width: 10 },
      { height: 10, id: 'y', width: 10 },
    ];
    const edges = [
      { sourceNodeId: 'p', targetNodeId: 'y' },
      { sourceNodeId: 'q', targetNodeId: 'x' },
    ];

    const positions = getSizedLayeredPositions(nodes, edges, { rowGap: 0 });

    expect(positions.y!.y).toBeLessThan(positions.x!.y);
    expect(positions.x!.x).toBe(positions.y!.x);
  });
});
