import { describe, expect, it } from 'vitest';

import type { LayoutGraphEdge, LayoutGraphNode } from './graphLayout';

import {
  getArrangedLayout,
  getLayeredPositions,
  getNodeDepths,
  getSizedLayeredPositions,
  getTopologicalOrder,
  UNMEASURED_ARRANGE_SIZE,
} from './graphLayout';

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
    // r -> p, r -> q, p -> y, q -> x: x and y are listed in the "wrong" order and swap rows to follow their parents.
    const nodes = [
      { height: 10, id: 'r', width: 10 },
      { height: 10, id: 'p', width: 10 },
      { height: 10, id: 'q', width: 10 },
      { height: 10, id: 'x', width: 10 },
      { height: 10, id: 'y', width: 10 },
    ];
    const edges = [
      { sourceNodeId: 'r', targetNodeId: 'p' },
      { sourceNodeId: 'r', targetNodeId: 'q' },
      { sourceNodeId: 'p', targetNodeId: 'y' },
      { sourceNodeId: 'q', targetNodeId: 'x' },
    ];

    const positions = getSizedLayeredPositions(nodes, edges, { rowGap: 0 });

    expect(positions.y!.y).toBeLessThan(positions.x!.y);
    expect(positions.x!.x).toBe(positions.y!.x);
  });

  it('packs unconnected nodes into a near-square grid instead of one column', () => {
    const nodes = Array.from({ length: 9 }, (_, index) => ({ height: 50, id: `n${index}`, width: 100 }));

    const positions = getSizedLayeredPositions(nodes, [], { layerGap: 10, rowGap: 10 });
    const xs = new Set(Object.values(positions).map((position) => position.x));
    const ys = new Set(Object.values(positions).map((position) => position.y));

    expect(xs).toEqual(new Set([0, 110, 220]));
    expect(ys).toEqual(new Set([0, 60, 120]));
    expect(positions.n0).toEqual({ x: 0, y: 0 });
    expect(positions.n3).toEqual({ x: 0, y: 60 });
  });

  it('lays out each connected component separately and shares a row while components fit', () => {
    const nodes = [
      { height: 50, id: 'a', width: 100 },
      { height: 50, id: 'b', width: 100 },
      { height: 50, id: 'lone', width: 100 },
    ];
    const edges = [{ sourceNodeId: 'a', targetNodeId: 'b' }];

    expect(getSizedLayeredPositions(nodes, edges, { layerGap: 10, rowGap: 10 })).toEqual({
      a: { x: 0, y: 0 },
      b: { x: 110, y: 0 },
      lone: { x: 220, y: 0 },
    });
  });

  it('gives a component wider than the row budget its own row', () => {
    const nodes = [
      { height: 50, id: 'wide', width: 500 },
      { height: 50, id: 'p', width: 100 },
      { height: 50, id: 'q', width: 100 },
    ];

    expect(getSizedLayeredPositions(nodes, [], { layerGap: 10, rowGap: 10 })).toEqual({
      wide: { x: 0, y: 0 },
      p: { x: 0, y: 60 },
      q: { x: 110, y: 60 },
    });
  });

  it('ignores edges that reference nodes outside the arranged set', () => {
    const nodes = [
      { height: 50, id: 'a', width: 100 },
      { height: 50, id: 'b', width: 100 },
    ];
    const edges = [
      { sourceNodeId: 'outside', targetNodeId: 'b' },
      { sourceNodeId: 'a', targetNodeId: 'b' },
    ];

    expect(getSizedLayeredPositions(nodes, edges, { layerGap: 10, rowGap: 10 })).toEqual({
      a: { x: 0, y: 0 },
      b: { x: 110, y: 0 },
    });
  });
});

describe('getArrangedLayout', () => {
  const sizes = (entries: Record<string, { height: number; width: number }>) => new Map(Object.entries(entries));

  it('arranges only the selection, anchored at its top-left corner, and leaves other nodes alone', () => {
    const nodes = [
      { id: 'a', position: { x: 400, y: 300 } },
      { id: 'b', position: { x: 900, y: 200 } },
      { id: 'outside', position: { x: 0, y: 0 } },
    ];
    const edges = [
      { source: 'a', target: 'b', type: 'default' },
      { source: 'outside', target: 'a', type: 'default' },
    ];
    const { bounds, positions } = getArrangedLayout(
      nodes,
      edges,
      new Set(['a', 'b']),
      sizes({ a: { height: 50, width: 100 }, b: { height: 60, width: 120 } })
    );

    expect(positions).toEqual({ a: { x: 400, y: 200 }, b: { x: 580, y: 200 } });
    expect(bounds).toEqual({ height: 60, width: 300, x: 400, y: 200 });
  });

  it('ranks by data edges only, so a loop linkage does not fold the body onto its head', () => {
    const nodes = [
      { id: 'head', position: { x: 0, y: 0 } },
      { id: 'body', position: { x: 0, y: 0 } },
    ];
    const edges = [
      { source: 'head', target: 'body', type: 'default' },
      { source: 'body', target: 'head', type: 'loop_linkage' },
    ];
    const { positions } = getArrangedLayout(
      nodes,
      edges,
      null,
      sizes({ body: { height: 50, width: 100 }, head: { height: 50, width: 100 } })
    );

    expect(positions.head!.x).toBeLessThan(positions.body!.x);
  });

  it('sizes a node the view has not measured at a typical node size', () => {
    const { bounds } = getArrangedLayout(
      [{ id: 'a', position: { x: 10, y: 20 } }],
      [],
      null,
      new Map([['a', { height: 0, width: 0 }]])
    );

    expect(bounds).toEqual({ ...UNMEASURED_ARRANGE_SIZE, x: 10, y: 20 });
  });
});
