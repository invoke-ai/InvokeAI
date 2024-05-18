import { deepClone } from 'common/util/deepClone';
import type { PendingConnection } from 'features/nodes/store/types';
import {
  getFirstValidConnection,
  getSourceCandidateFields,
  getTargetCandidateFields,
} from 'features/nodes/store/util/getFirstValidConnection';
import { add, buildEdge, buildNode, img_resize, templates } from 'features/nodes/store/util/testUtils';
import { unset } from 'lodash-es';
import { describe, expect, it } from 'vitest';

describe('getFirstValidConnection', () => {
  it('should return null if the pending and candidate nodes are the same node', () => {
    const pc: PendingConnection = { node: buildNode(add), template: add, fieldTemplate: add.inputs['a']! };
    const candidateNode = pc.node;
    expect(getFirstValidConnection(templates, [pc.node], [], pc, candidateNode, add, null)).toBe(null);
  });

  describe('connecting from a source to a target', () => {
    const pc: PendingConnection = {
      node: buildNode(img_resize),
      template: img_resize,
      fieldTemplate: img_resize.outputs['width']!,
    };
    const candidateNode = buildNode(img_resize);

    it('should return the first valid connection if there are no connected fields', () => {
      const r = getFirstValidConnection(templates, [pc.node, candidateNode], [], pc, candidateNode, img_resize, null);
      const c = {
        source: pc.node.id,
        sourceHandle: pc.fieldTemplate.name,
        target: candidateNode.id,
        targetHandle: 'width',
      };
      expect(r).toEqual(c);
    });
    it('should return the first valid connection if there is a connected field', () => {
      const r = getFirstValidConnection(
        templates,
        [pc.node, candidateNode],
        [buildEdge(pc.node.id, 'width', candidateNode.id, 'width')],
        pc,
        candidateNode,
        img_resize,
        null
      );
      const c = {
        source: pc.node.id,
        sourceHandle: pc.fieldTemplate.name,
        target: candidateNode.id,
        targetHandle: 'height',
      };
      expect(r).toEqual(c);
    });
    it('should return the first valid connection if there is an edgePendingUpdate', () => {
      const e = buildEdge(pc.node.id, 'width', candidateNode.id, 'width');
      const r = getFirstValidConnection(templates, [pc.node, candidateNode], [e], pc, candidateNode, img_resize, e);
      const c = {
        source: pc.node.id,
        sourceHandle: pc.fieldTemplate.name,
        target: candidateNode.id,
        targetHandle: 'width',
      };
      expect(r).toEqual(c);
    });
  });
  describe('connecting from a target to a source', () => {});
});

describe('getTargetCandidateFields', () => {
  it('should return an empty array if the nodes canot be found', () => {
    const r = getTargetCandidateFields('missing', 'value', 'missing', [], [], templates, null);
    expect(r).toEqual([]);
  });
  it('should return an empty array if the templates cannot be found', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(add);
    const nodes = [n1, n2];
    const r = getTargetCandidateFields(n1.id, 'value', n2.id, nodes, [], {}, null);
    expect(r).toEqual([]);
  });
  it('should return an empty array if the source field template cannot be found', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(add);
    const nodes = [n1, n2];

    const addWithoutOutputValue = deepClone(add);
    unset(addWithoutOutputValue, 'outputs.value');

    const r = getTargetCandidateFields(n1.id, 'value', n2.id, nodes, [], { add: addWithoutOutputValue }, null);
    expect(r).toEqual([]);
  });
  it('should return all valid target fields if there are no connected fields', () => {
    const n1 = buildNode(img_resize);
    const n2 = buildNode(img_resize);
    const nodes = [n1, n2];
    const r = getTargetCandidateFields(n1.id, 'width', n2.id, nodes, [], templates, null);
    expect(r).toEqual([img_resize.inputs['width'], img_resize.inputs['height']]);
  });
  it('should ignore the edgePendingUpdate if provided', () => {
    const n1 = buildNode(img_resize);
    const n2 = buildNode(img_resize);
    const nodes = [n1, n2];
    const edgePendingUpdate = buildEdge(n1.id, 'width', n2.id, 'width');
    const r = getTargetCandidateFields(n1.id, 'width', n2.id, nodes, [], templates, edgePendingUpdate);
    expect(r).toEqual([img_resize.inputs['width'], img_resize.inputs['height']]);
  });
});

describe('getSourceCandidateFields', () => {
  it('should return an empty array if the nodes canot be found', () => {
    const r = getSourceCandidateFields('missing', 'value', 'missing', [], [], templates, null);
    expect(r).toEqual([]);
  });
  it('should return an empty array if the templates cannot be found', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(add);
    const nodes = [n1, n2];
    const r = getSourceCandidateFields(n2.id, 'a', n1.id, nodes, [], {}, null);
    expect(r).toEqual([]);
  });
  it('should return an empty array if the source field template cannot be found', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(add);
    const nodes = [n1, n2];

    const addWithoutInputA = deepClone(add);
    unset(addWithoutInputA, 'inputs.a');

    const r = getSourceCandidateFields(n1.id, 'a', n2.id, nodes, [], { add: addWithoutInputA }, null);
    expect(r).toEqual([]);
  });
  it('should return all valid source fields if there are no connected fields', () => {
    const n1 = buildNode(img_resize);
    const n2 = buildNode(img_resize);
    const nodes = [n1, n2];
    const r = getSourceCandidateFields(n2.id, 'width', n1.id, nodes, [], templates, null);
    expect(r).toEqual([img_resize.outputs['width'], img_resize.outputs['height']]);
  });
  it('should ignore the edgePendingUpdate if provided', () => {
    const n1 = buildNode(img_resize);
    const n2 = buildNode(img_resize);
    const nodes = [n1, n2];
    const edgePendingUpdate = buildEdge(n1.id, 'width', n2.id, 'width');
    const r = getSourceCandidateFields(n2.id, 'width', n1.id, nodes, [], templates, edgePendingUpdate);
    expect(r).toEqual([img_resize.outputs['width'], img_resize.outputs['height']]);
  });
});
