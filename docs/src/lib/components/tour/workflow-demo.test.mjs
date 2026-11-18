import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  EDGE_INSET,
  WORLD,
  WORLD_BOUNDS,
  canConnect,
  clampPosition,
  connectEdge,
  createInitialGraph,
  findNearest,
  getConnectedInputs,
  getEdgePath,
  getHandlePoint,
  layoutNode,
  toGraphBounds,
} from './workflow-demo.mjs';

const nodeById = (id) => createInitialGraph().nodes.find((node) => node.id === id);

describe('getEdgePath', () => {
  it('bends forward edges halfway across the gap, as xyflow does', () => {
    assert.equal(getEdgePath({ x: 100, y: 50 }, { x: 300, y: 150 }), 'M100,50 C200,50 200,150 300,150');
  });

  it('loops a target behind its source out and back by 6.25 * sqrt(distance)', () => {
    // 0.25 curvature * 25 * sqrt(200) = 88.39
    assert.equal(getEdgePath({ x: 300, y: 50 }, { x: 100, y: 150 }), 'M300,50 C388.39,50 11.61,150 100,150');
  });
});

describe('clampPosition', () => {
  const size = { width: 200, height: 100 };

  it('keeps a position that is already inside', () => {
    assert.deepEqual(clampPosition({ x: 40, y: 30 }, size, WORLD_BOUNDS), { x: 40, y: 30 });
  });

  it('pulls a node back inside every edge', () => {
    assert.deepEqual(clampPosition({ x: -20, y: -5 }, size, WORLD_BOUNDS), { x: 0, y: 0 });
    assert.deepEqual(clampPosition({ x: 900, y: 900 }, size, WORLD_BOUNDS), { x: 480, y: 340 });
  });

  it('pins a node larger than the bounds to the top left, keeping its header reachable', () => {
    assert.deepEqual(clampPosition({ x: 50, y: 50 }, { width: 800, height: 500 }, WORLD_BOUNDS), { x: 0, y: 0 });
  });

  it('accepts bounds that extend past the graph on every side', () => {
    const section = { left: -400, top: -60, right: 680, bottom: 500 };
    assert.deepEqual(clampPosition({ x: -900, y: -900 }, size, section, 6), { x: -394, y: -54 });
    assert.deepEqual(clampPosition({ x: -250, y: 20 }, size, section, 6), { x: -250, y: 20 });
  });

  it('keeps the inset clear on every side', () => {
    assert.deepEqual(clampPosition({ x: -20, y: -5 }, size, WORLD_BOUNDS, 6), { x: 6, y: 6 });
    assert.deepEqual(clampPosition({ x: 900, y: 900 }, size, WORLD_BOUNDS, 6), { x: 474, y: 334 });
  });
});

describe('toGraphBounds', () => {
  it('maps a surrounding area into graph coordinates at the scale the graph is drawn at', () => {
    // The graph is drawn at half size (340px wide) 200px right of and 30px below the section's top left corner.
    const stage = { left: 300, top: 130, right: 640, bottom: 350, width: 340 };
    const section = { left: 100, top: 100, right: 660, bottom: 400 };
    assert.deepEqual(toGraphBounds(section, stage), { left: -400, top: -60, right: 720, bottom: 540 });
  });
});

describe('layoutNode', () => {
  it('stacks outputs, then inputs, with connected inputs hiding their controls', () => {
    const { edges } = createInitialGraph();
    const layout = layoutNode(nodeById('compositor'), getConnectedInputs(edges, 'compositor'));
    // 1px border + 36px header + 4px body padding, then 24px rows centered at 12px.
    assert.deepEqual(layout.outputs, { image: 53, width: 77, height: 101 });
    // Board (no handle) takes 24 + 2 + 28; Image Subject is connected; Image Background shows its 118px image control.
    assert.deepEqual(layout.inputs, { subject: 179, background: 203, threshold: 347 });
    // Threshold is connected (24) + 4px padding + 32px footer + 1px border.
    assert.equal(layout.height, 396);
  });

  it('grows by the control height when an input is disconnected', () => {
    const connected = layoutNode(nodeById('compositor'), new Set(['subject', 'threshold']));
    const disconnected = layoutNode(nodeById('compositor'), new Set(['subject']));
    assert.equal(disconnected.height - connected.height, 30);
  });

  it('collapses to the header and gathers every handle behind it', () => {
    const node = { ...nodeById('l2i'), open: false };
    const layout = layoutNode(node, new Set());
    assert.equal(layout.height, 38);
    assert.deepEqual(new Set([...Object.values(layout.inputs), ...Object.values(layout.outputs)]), new Set([23]));
    assert.deepEqual(Object.keys(layout.inputs), ['latents', 'vae', 'tiled']);
  });

  it('places inputs on the left edge and outputs on the right edge of the node', () => {
    const node = nodeById('integer');
    const layout = layoutNode(node, new Set());
    assert.deepEqual(getHandlePoint(node, layout, 'input', 'value'), { x: node.x, y: node.y + 77 });
    assert.deepEqual(getHandlePoint(node, layout, 'output', 'value'), { x: node.x + node.width, y: node.y + 53 });
    assert.equal(getHandlePoint(node, layout, 'input', 'missing'), null);
  });
});

describe('the initial graph', () => {
  it('fits on the canvas with every input disconnected, so rewiring never pushes a node off it', () => {
    for (const node of createInitialGraph().nodes) {
      const { height } = layoutNode(node, new Set());
      const position = clampPosition(node, { width: node.width, height }, WORLD_BOUNDS, EDGE_INSET);
      assert.deepEqual(position, { x: node.x, y: node.y }, node.id);
    }
  });

  it('connects only fields that accept each other', () => {
    const { nodes, edges } = createInitialGraph();
    for (const edge of edges) {
      assert.ok(canConnect(nodes, { node: edge.source, handle: edge.sourceHandle }, { node: edge.target, handle: edge.targetHandle }));
    }
  });
});

describe('canConnect', () => {
  const { nodes } = createInitialGraph();

  it('accepts an output feeding an input of the same field type on another node', () => {
    assert.ok(canConnect(nodes, { node: 'l2i', handle: 'width' }, { node: 'compositor', handle: 'threshold' }));
  });

  it('rejects mismatched types, inputs without a handle and loops onto the same node', () => {
    assert.equal(canConnect(nodes, { node: 'l2i', handle: 'image' }, { node: 'compositor', handle: 'threshold' }), false);
    assert.equal(canConnect(nodes, { node: 'l2i', handle: 'image' }, { node: 'compositor', handle: 'board' }), false);
    assert.equal(canConnect(nodes, { node: 'integer', handle: 'value' }, { node: 'integer', handle: 'value' }), false);
  });
});

describe('connectEdge', () => {
  it('replaces the edge already feeding the input and keeps the others', () => {
    const { edges } = createInitialGraph();
    const edge = { source: 'l2i', sourceHandle: 'height', target: 'compositor', targetHandle: 'threshold' };
    assert.deepEqual(connectEdge(edges, edge), [edges[0], edge]);
  });
});

describe('findNearest', () => {
  const candidates = [
    { id: 'a', x: 0, y: 0 },
    { id: 'b', x: 30, y: 0 },
  ];

  it('snaps to the nearest candidate within the radius', () => {
    assert.equal(findNearest(candidates, { x: 18, y: 4 }, 20)?.id, 'b');
  });

  it('finds nothing when every candidate is beyond the radius', () => {
    assert.equal(findNearest(candidates, { x: 15, y: 30 }, 20), null);
  });
});
