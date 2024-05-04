import { Graph } from 'features/nodes/util/graph/Graph';
import type { Invocation } from 'services/api/types';
import { assert, AssertionError, is } from 'tsafe';
import { validate } from 'uuid';
import { describe, expect, it } from 'vitest';

describe('Graph', () => {
  describe('constructor', () => {
    it('should create a new graph with the correct id', () => {
      const g = new Graph('test-id');
      expect(g._graph.id).toBe('test-id');
    });
    it('should create a new graph with a uuid id if none is provided', () => {
      const g = new Graph();
      expect(g._graph.id).not.toBeUndefined();
      expect(validate(g._graph.id)).toBeTruthy();
    });
  });

  describe('addNode', () => {
    const testNode = {
      id: 'test-node',
      type: 'add',
    } as const;
    it('should add a node to the graph', () => {
      const g = new Graph();
      g.addNode(testNode);
      expect(g._graph.nodes['test-node']).not.toBeUndefined();
      expect(g._graph.nodes['test-node']?.type).toBe('add');
    });
    it('should set is_intermediate to true if not provided', () => {
      const g = new Graph();
      g.addNode(testNode);
      expect(g._graph.nodes['test-node']?.is_intermediate).toBe(true);
    });
    it('should not overwrite is_intermediate if provided', () => {
      const g = new Graph();
      g.addNode({
        ...testNode,
        is_intermediate: false,
      });
      expect(g._graph.nodes['test-node']?.is_intermediate).toBe(false);
    });
    it('should set use_cache to true if not provided', () => {
      const g = new Graph();
      g.addNode(testNode);
      expect(g._graph.nodes['test-node']?.use_cache).toBe(true);
    });
    it('should not overwrite use_cache if provided', () => {
      const g = new Graph();
      g.addNode({
        ...testNode,
        use_cache: false,
      });
      expect(g._graph.nodes['test-node']?.use_cache).toBe(false);
    });
    it('should error if the node id is already in the graph', () => {
      const g = new Graph();
      g.addNode(testNode);
      expect(() => g.addNode(testNode)).toThrowError(AssertionError);
    });
    it('should infer the types if provided', () => {
      const g = new Graph();
      const node = g.addNode(testNode);
      assert(is<Invocation<'add'>>(node));
      const g2 = new Graph();
      // @ts-expect-error The node object is an `add` type, but the generic is a `sub` type
      g2.addNode<'sub'>(testNode);
    });
  });

  describe('updateNode', () => {
    it('should update the node with the provided id', () => {
      const g = new Graph();
      const node: Invocation<'add'> = {
        id: 'test-node',
        type: 'add',
        a: 1,
      };
      g.addNode(node);
      const updatedNode = g.updateNode('test-node', 'add', {
        a: 2,
      });
      expect(g.getNode('test-node', 'add').a).toBe(2);
      expect(node).toBe(updatedNode);
    });
    it('should throw an error if the node is not found', () => {
      expect(() => new Graph().updateNode('not-found', 'add', {})).toThrowError(AssertionError);
    });
    it('should throw an error if the node is found but has the wrong type', () => {
      const g = new Graph();
      g.addNode({
        id: 'test-node',
        type: 'add',
        a: 1,
      });
      expect(() => g.updateNode('test-node', 'sub', {})).toThrowError(AssertionError);
    });
    it('should infer types correctly when `type` is omitted', () => {
      const g = new Graph();
      g.addNode({
        id: 'test-node',
        type: 'add',
        a: 1,
      });
      const updatedNode = g.updateNode('test-node', 'add', {
        a: 2,
      });
      assert(is<Invocation<'add'>>(updatedNode));
    });
    it('should infer types correctly when `type` is provided', () => {
      const g = new Graph();
      g.addNode({
        id: 'test-node',
        type: 'add',
        a: 1,
      });
      const updatedNode = g.updateNode('test-node', 'add', {
        a: 2,
      });
      assert(is<Invocation<'add'>>(updatedNode));
    });
  });

  describe('addEdge', () => {
    it('should add an edge to the graph with the provided values', () => {
      const g = new Graph();
      g.addEdge<'add', 'sub'>('from-node', 'value', 'to-node', 'b');
      expect(g._graph.edges.length).toBe(1);
      expect(g._graph.edges[0]).toEqual({
        source: { node_id: 'from-node', field: 'value' },
        destination: { node_id: 'to-node', field: 'b' },
      });
    });
    it('should throw an error if the edge already exists', () => {
      const g = new Graph();
      g.addEdge<'add', 'sub'>('from-node', 'value', 'to-node', 'b');
      expect(() => g.addEdge<'add', 'sub'>('from-node', 'value', 'to-node', 'b')).toThrowError(AssertionError);
    });
    it('should infer field names', () => {
      const g = new Graph();
      // @ts-expect-error The first field must be a valid output field of the first type arg
      g.addEdge<'add', 'sub'>('from-node', 'not-a-valid-field', 'to-node', 'a');
      // @ts-expect-error The second field must be a valid input field of the second type arg
      g.addEdge<'add', 'sub'>('from-node-2', 'value', 'to-node-2', 'not-a-valid-field');
      // @ts-expect-error The first field must be any valid output field
      g.addEdge('from-node-3', 'not-a-valid-field', 'to-node-3', 'a');
      // @ts-expect-error The second field must be any valid input field
      g.addEdge('from-node-4', 'clip', 'to-node-4', 'not-a-valid-field');
    });
  });

  describe('getNode', () => {
    const g = new Graph();
    const node = g.addNode({
      id: 'test-node',
      type: 'add',
    });

    it('should return the node with the provided id', () => {
      const n = g.getNode('test-node');
      expect(n).toBe(node);
    });
    it('should return the node with the provided id and type', () => {
      const n = g.getNode('test-node', 'add');
      expect(n).toBe(node);
      assert(is<Invocation<'add'>>(node));
    });
    it('should throw an error if the node is not found', () => {
      expect(() => g.getNode('not-found')).toThrowError(AssertionError);
    });
    it('should throw an error if the node is found but has the wrong type', () => {
      expect(() => g.getNode('test-node', 'sub')).toThrowError(AssertionError);
    });
  });

  describe('getNodeSafe', () => {
    const g = new Graph();
    const node = g.addNode({
      id: 'test-node',
      type: 'add',
    });
    it('should return the node if it is found', () => {
      expect(g.getNodeSafe('test-node')).toBe(node);
    });
    it('should return the node if it is found with the provided type', () => {
      expect(g.getNodeSafe('test-node')).toBe(node);
      assert(is<Invocation<'add'>>(node));
    });
    it("should return undefined if the node isn't found", () => {
      expect(g.getNodeSafe('not-found')).toBeUndefined();
    });
    it('should return undefined if the node is found but has the wrong type', () => {
      expect(g.getNodeSafe('test-node', 'sub')).toBeUndefined();
    });
  });

  describe('hasNode', () => {
    const g = new Graph();
    g.addNode({
      id: 'test-node',
      type: 'add',
    });

    it('should return true if the node is in the graph', () => {
      expect(g.hasNode('test-node')).toBe(true);
    });
    it('should return false if the node is not in the graph', () => {
      expect(g.hasNode('not-found')).toBe(false);
    });
  });

  describe('getEdge', () => {
    const g = new Graph();
    g.addEdge<'add', 'sub'>('from-node', 'value', 'to-node', 'b');
    it('should return the edge with the provided values', () => {
      expect(g.getEdge('from-node', 'value', 'to-node', 'b')).toEqual({
        source: { node_id: 'from-node', field: 'value' },
        destination: { node_id: 'to-node', field: 'b' },
      });
    });
    it('should throw an error if the edge is not found', () => {
      expect(() => g.getEdge('from-node', 'value', 'to-node', 'a')).toThrowError(AssertionError);
    });
  });

  describe('getEdgeSafe', () => {
    const g = new Graph();
    g.addEdge<'add', 'sub'>('from-node', 'value', 'to-node', 'b');
    it('should return the edge if it is found', () => {
      expect(g.getEdgeSafe('from-node', 'value', 'to-node', 'b')).toEqual({
        source: { node_id: 'from-node', field: 'value' },
        destination: { node_id: 'to-node', field: 'b' },
      });
    });
    it('should return undefined if the edge is not found', () => {
      expect(g.getEdgeSafe('from-node', 'value', 'to-node', 'a')).toBeUndefined();
    });
  });

  describe('hasEdge', () => {
    const g = new Graph();
    g.addEdge<'add', 'sub'>('from-node', 'value', 'to-node', 'b');
    it('should return true if the edge is in the graph', () => {
      expect(g.hasEdge('from-node', 'value', 'to-node', 'b')).toBe(true);
    });
    it('should return false if the edge is not in the graph', () => {
      expect(g.hasEdge('from-node', 'value', 'to-node', 'a')).toBe(false);
    });
  });

  describe('getGraph', () => {
    it('should return the graph', () => {
      const g = new Graph();
      expect(g.getGraph()).toBe(g._graph);
    });
    it('should raise an error if the graph is invalid', () => {
      const g = new Graph();
      g.addEdge('from-node', 'value', 'to-node', 'b');
      expect(() => g.getGraph()).toThrowError(AssertionError);
    });
  });

  describe('getGraphSafe', () => {
    it('should return the graph even if it is invalid', () => {
      const g = new Graph();
      g.addEdge('from-node', 'value', 'to-node', 'b');
      expect(g.getGraphSafe()).toBe(g._graph);
    });
  });

  describe('validate', () => {
    it('should not throw an error if the graph is valid', () => {
      const g = new Graph();
      expect(() => g.validate()).not.toThrow();
    });
    it('should throw an error if the graph is invalid', () => {
      const g = new Graph();
      // edge from nowhere to nowhere
      g.addEdge('from-node', 'value', 'to-node', 'b');
      expect(() => g.validate()).toThrowError(AssertionError);
    });
  });

  describe('traversal', () => {
    const g = new Graph();
    const n1 = g.addNode({
      id: 'n1',
      type: 'add',
    });
    const n2 = g.addNode({
      id: 'n2',
      type: 'alpha_mask_to_tensor',
    });
    const n3 = g.addNode({
      id: 'n3',
      type: 'add',
    });
    const n4 = g.addNode({
      id: 'n4',
      type: 'add',
    });
    const n5 = g.addNode({
      id: 'n5',
      type: 'add',
    });
    const e1 = g.addEdge<'add', 'add'>(n1.id, 'value', n3.id, 'a');
    const e2 = g.addEdge<'alpha_mask_to_tensor', 'add'>(n2.id, 'height', n3.id, 'b');
    const e3 = g.addEdge<'add', 'add'>(n3.id, 'value', n4.id, 'a');
    const e4 = g.addEdge<'add', 'add'>(n3.id, 'value', n5.id, 'b');
    describe('getEdgesFrom', () => {
      it('should return the edges that start at the provided node', () => {
        expect(g.getEdgesFrom(n3.id)).toEqual([e3, e4]);
      });
      it('should return the edges that start at the provided node and have the provided field', () => {
        expect(g.getEdgesFrom(n2.id, 'height')).toEqual([e2]);
      });
    });
    describe('getEdgesTo', () => {
      it('should return the edges that end at the provided node', () => {
        expect(g.getEdgesTo(n3.id)).toEqual([e1, e2]);
      });
      it('should return the edges that end at the provided node and have the provided field', () => {
        expect(g.getEdgesTo(n3.id, 'b')).toEqual([e2]);
      });
    });
    describe('getIncomers', () => {
      it('should return the nodes that have an edge to the provided node', () => {
        expect(g.getIncomers(n3.id)).toEqual([n1, n2]);
      });
    });
    describe('getOutgoers', () => {
      it('should return the nodes that the provided node has an edge to', () => {
        expect(g.getOutgoers(n3.id)).toEqual([n4, n5]);
      });
    });
  });
});
