import { isModelIdentifier } from 'features/nodes/types/common';
import { Graph } from 'features/nodes/util/graph/Graph';
import { MetadataUtil } from 'features/nodes/util/graph/MetadataUtil';
import { pick } from 'lodash-es';
import type { AnyModelConfig } from 'services/api/types';
import { AssertionError } from 'tsafe';
import { describe, expect, it } from 'vitest';

describe('MetadataUtil', () => {
  describe('getNode', () => {
    it('should return the metadata node if one exists', () => {
      const g = new Graph();
      // @ts-expect-error `Graph` excludes `core_metadata` nodes due to its excessively wide typing
      const metadataNode = g.addNode({ id: MetadataUtil.metadataNodeId, type: 'core_metadata' });
      expect(MetadataUtil.getNode(g)).toEqual(metadataNode);
    });
    it('should raise an error if the metadata node does not exist', () => {
      const g = new Graph();
      expect(() => MetadataUtil.getNode(g)).toThrowError(AssertionError);
    });
  });

  describe('add', () => {
    const g = new Graph();
    it("should add metadata, creating the node if it doesn't exist", () => {
      MetadataUtil.add(g, { foo: 'bar' });
      const metadataNode = MetadataUtil.getNode(g);
      expect(metadataNode['type']).toBe('core_metadata');
      expect(metadataNode['foo']).toBe('bar');
    });
    it('should update existing metadata keys', () => {
      const updatedMetadataNode = MetadataUtil.add(g, { foo: 'bananas', baz: 'qux' });
      expect(updatedMetadataNode['foo']).toBe('bananas');
      expect(updatedMetadataNode['baz']).toBe('qux');
    });
  });

  describe('remove', () => {
    it('should remove a single key', () => {
      const g = new Graph();
      MetadataUtil.add(g, { foo: 'bar', baz: 'qux' });
      const updatedMetadataNode = MetadataUtil.remove(g, 'foo');
      expect(updatedMetadataNode['foo']).toBeUndefined();
      expect(updatedMetadataNode['baz']).toBe('qux');
    });
    it('should remove multiple keys', () => {
      const g = new Graph();
      MetadataUtil.add(g, { foo: 'bar', baz: 'qux' });
      const updatedMetadataNode = MetadataUtil.remove(g, ['foo', 'baz']);
      expect(updatedMetadataNode['foo']).toBeUndefined();
      expect(updatedMetadataNode['baz']).toBeUndefined();
    });
  });

  describe('setMetadataReceivingNode', () => {
    const g = new Graph();
    it('should add an edge from from metadata to the receiving node', () => {
      const n = g.addNode({ id: 'my-node', type: 'img_resize' });
      MetadataUtil.add(g, { foo: 'bar' });
      MetadataUtil.setMetadataReceivingNode(g, n);
      // @ts-expect-error `Graph` excludes `core_metadata` nodes due to its excessively wide typing
      expect(g.hasEdge(MetadataUtil.getNode(g), 'metadata', n, 'metadata')).toBe(true);
    });
    it('should remove existing metadata edges', () => {
      const n2 = g.addNode({ id: 'my-other-node', type: 'img_resize' });
      MetadataUtil.setMetadataReceivingNode(g, n2);
      expect(g.getIncomers(n2).length).toBe(1);
      // @ts-expect-error `Graph` excludes `core_metadata` nodes due to its excessively wide typing
      expect(g.hasEdge(MetadataUtil.getNode(g), 'metadata', n2, 'metadata')).toBe(true);
    });
  });

  describe('getModelMetadataField', () => {
    it('should return a ModelIdentifierField', () => {
      const model: AnyModelConfig = {
        key: 'model_key',
        type: 'main',
        hash: 'model_hash',
        base: 'sd-1',
        format: 'diffusers',
        name: 'my model',
        path: '/some/path',
        source: 'www.models.com',
        source_type: 'url',
      };
      const metadataField = MetadataUtil.getModelMetadataField(model);
      expect(isModelIdentifier(metadataField)).toBe(true);
      expect(pick(model, ['key', 'hash', 'name', 'base', 'type'])).toEqual(metadataField);
    });
  });
});
