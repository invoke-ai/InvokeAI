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

  describe('addEdge', () => {
    const add: Invocation<'add'> = {
      id: 'from-node',
      type: 'add',
    };
    const sub: Invocation<'sub'> = {
      id: 'to-node',
      type: 'sub',
    };
    it('should add an edge to the graph with the provided values', () => {
      const g = new Graph();
      g.addNode(add);
      g.addNode(sub);
      g.addEdge(add, 'value', sub, 'b');
      expect(g._graph.edges.length).toBe(1);
      expect(g._graph.edges[0]).toEqual({
        source: { node_id: 'from-node', field: 'value' },
        destination: { node_id: 'to-node', field: 'b' },
      });
    });
    it('should throw an error if the edge already exists', () => {
      const g = new Graph();
      g.addEdge(add, 'value', sub, 'b');
      expect(() => g.addEdge(add, 'value', sub, 'b')).toThrowError(AssertionError);
    });
    it('should infer field names', () => {
      const g = new Graph();
      // @ts-expect-error The first field must be a valid output field of the first type arg
      g.addEdge(add, 'not-a-valid-field', add, 'a');
      // @ts-expect-error The second field must be a valid input field of the second type arg
      g.addEdge(add, 'value', sub, 'not-a-valid-field');
      // @ts-expect-error The first field must be any valid output field
      g.addEdge(add, 'not-a-valid-field', sub, 'a');
      // @ts-expect-error The second field must be any valid input field
      g.addEdge(add, 'clip', sub, 'not-a-valid-field');
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
    it('should throw an error if the node is not found', () => {
      expect(() => g.getNode('not-found')).toThrowError(AssertionError);
    });
  });

  describe('deleteNode', () => {
    it('should delete the node with the provided id', () => {
      const g = new Graph();
      const n1 = g.addNode({
        id: 'n1',
        type: 'add',
      });
      const n2 = g.addNode({
        id: 'n2',
        type: 'add',
      });
      const n3 = g.addNode({
        id: 'n3',
        type: 'add',
      });
      g.addEdge(n1, 'value', n2, 'a');
      g.addEdge(n2, 'value', n3, 'a');
      // This edge should not be deleted bc it doesn't touch n2
      g.addEdge(n1, 'value', n3, 'a');
      g.deleteNode(n2.id);
      expect(g.hasNode(n1.id)).toBe(true);
      expect(g.hasNode(n2.id)).toBe(false);
      expect(g.hasNode(n3.id)).toBe(true);
      // Should delete edges to and from the node
      expect(g.getEdges().length).toBe(1);
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
    const add: Invocation<'add'> = {
      id: 'from-node',
      type: 'add',
    };
    const sub: Invocation<'sub'> = {
      id: 'to-node',
      type: 'sub',
    };
    g.addEdge(add, 'value', sub, 'b');
    it('should return the edge with the provided values', () => {
      expect(g.getEdge(add, 'value', sub, 'b')).toEqual({
        source: { node_id: 'from-node', field: 'value' },
        destination: { node_id: 'to-node', field: 'b' },
      });
    });
    it('should throw an error if the edge is not found', () => {
      expect(() => g.getEdge(add, 'value', sub, 'a')).toThrowError(AssertionError);
    });
  });

  describe('getEdges', () => {
    it('should get all edges in the graph', () => {
      const g = new Graph();
      const n1 = g.addNode({
        id: 'n1',
        type: 'add',
      });
      const n2 = g.addNode({
        id: 'n2',
        type: 'add',
      });
      const n3 = g.addNode({
        id: 'n3',
        type: 'add',
      });
      const e1 = g.addEdge(n1, 'value', n2, 'a');
      const e2 = g.addEdge(n2, 'value', n3, 'a');
      expect(g.getEdges()).toEqual([e1, e2]);
    });
  });

  describe('hasEdge', () => {
    const g = new Graph();
    const add: Invocation<'add'> = {
      id: 'from-node',
      type: 'add',
    };
    const sub: Invocation<'sub'> = {
      id: 'to-node',
      type: 'sub',
    };
    g.addEdge(add, 'value', sub, 'b');
    it('should return true if the edge is in the graph', () => {
      expect(g.hasEdge(add, 'value', sub, 'b')).toBe(true);
    });
    it('should return false if the edge is not in the graph', () => {
      expect(g.hasEdge(add, 'value', sub, 'a')).toBe(false);
    });
  });

  describe('getGraph', () => {
    it('should return the graph', () => {
      const g = new Graph();
      expect(g.getGraph()).toBe(g._graph);
    });
    it('should raise an error if the graph is invalid', () => {
      const g = new Graph();
      const add: Invocation<'add'> = {
        id: 'from-node',
        type: 'add',
      };
      const sub: Invocation<'sub'> = {
        id: 'to-node',
        type: 'sub',
      };
      g.addEdge(add, 'value', sub, 'b');
      expect(() => g.getGraph()).toThrowError(AssertionError);
    });
  });

  describe('getGraphSafe', () => {
    it('should return the graph even if it is invalid', () => {
      const g = new Graph();
      const add: Invocation<'add'> = {
        id: 'from-node',
        type: 'add',
      };
      const sub: Invocation<'sub'> = {
        id: 'to-node',
        type: 'sub',
      };
      g.addEdge(add, 'value', sub, 'b');
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
      const add: Invocation<'add'> = {
        id: 'from-node',
        type: 'add',
      };
      const sub: Invocation<'sub'> = {
        id: 'to-node',
        type: 'sub',
      };
      // edge from nowhere to nowhere
      g.addEdge(add, 'value', sub, 'b');
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
    const e1 = g.addEdge(n1, 'value', n3, 'a');
    const e2 = g.addEdge(n2, 'height', n3, 'b');
    const e3 = g.addEdge(n3, 'value', n4, 'a');
    const e4 = g.addEdge(n3, 'value', n5, 'b');
    describe('getEdgesFrom', () => {
      it('should return the edges that start at the provided node', () => {
        expect(g.getEdgesFrom(n3)).toEqual([e3, e4]);
      });
      it('should return the edges that start at the provided node and have the provided field', () => {
        expect(g.getEdgesFrom(n2, 'height')).toEqual([e2]);
      });
    });
    describe('getEdgesTo', () => {
      it('should return the edges that end at the provided node', () => {
        expect(g.getEdgesTo(n3)).toEqual([e1, e2]);
      });
      it('should return the edges that end at the provided node and have the provided field', () => {
        expect(g.getEdgesTo(n3, 'b')).toEqual([e2]);
      });
    });
    describe('getIncomers', () => {
      it('should return the nodes that have an edge to the provided node', () => {
        expect(g.getIncomers(n3)).toEqual([n1, n2]);
      });
    });
    describe('getOutgoers', () => {
      it('should return the nodes that the provided node has an edge to', () => {
        expect(g.getOutgoers(n3)).toEqual([n4, n5]);
      });
    });
  });

  describe('deleteEdgesFrom', () => {
    it('should delete edges from the provided node', () => {
      const g = new Graph();
      const n1 = g.addNode({
        id: 'n1',
        type: 'img_resize',
      });
      const n2 = g.addNode({
        id: 'n2',
        type: 'add',
      });
      const _e1 = g.addEdge(n1, 'height', n2, 'a');
      const _e2 = g.addEdge(n1, 'width', n2, 'b');
      g.deleteEdgesFrom(n1);
      expect(g.getEdgesFrom(n1)).toEqual([]);
    });
    it('should delete edges from the provided node, with the provided field', () => {
      const g = new Graph();
      const n1 = g.addNode({
        id: 'n1',
        type: 'img_resize',
      });
      const n2 = g.addNode({
        id: 'n2',
        type: 'add',
      });
      const n3 = g.addNode({
        id: 'n3',
        type: 'add',
      });
      const _e1 = g.addEdge(n1, 'height', n2, 'a');
      const e2 = g.addEdge(n1, 'width', n2, 'b');
      const e3 = g.addEdge(n1, 'width', n3, 'b');
      g.deleteEdgesFrom(n1, 'height');
      expect(g.getEdgesFrom(n1)).toEqual([e2, e3]);
    });
  });

  describe('deleteEdgesTo', () => {
    it('should delete edges to the provided node', () => {
      const g = new Graph();
      const n1 = g.addNode({
        id: 'n1',
        type: 'img_resize',
      });
      const n2 = g.addNode({
        id: 'n2',
        type: 'add',
      });
      const _e1 = g.addEdge(n1, 'height', n2, 'a');
      const _e2 = g.addEdge(n1, 'width', n2, 'b');
      g.deleteEdgesTo(n2);
      expect(g.getEdgesTo(n2)).toEqual([]);
    });
    it('should delete edges to the provided node, with the provided field', () => {
      const g = new Graph();
      const n1 = g.addNode({
        id: 'n1',
        type: 'img_resize',
      });
      const n2 = g.addNode({
        id: 'n2',
        type: 'img_resize',
      });
      const n3 = g.addNode({
        id: 'n3',
        type: 'add',
      });
      const _e1 = g.addEdge(n1, 'height', n3, 'a');
      const e2 = g.addEdge(n1, 'width', n3, 'b');
      const _e3 = g.addEdge(n2, 'width', n3, 'a');
      g.deleteEdgesTo(n3, 'a');
      expect(g.getEdgesTo(n3)).toEqual([e2]);
    });
  });
});
