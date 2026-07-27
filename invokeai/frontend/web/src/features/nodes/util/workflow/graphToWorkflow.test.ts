import { $templates } from 'features/nodes/store/nodesSlice';
import type { InvocationTemplate } from 'features/nodes/types/invocation';
import { isWorkflowInvocationNode } from 'features/nodes/types/workflow';
import { graphToWorkflow } from 'features/nodes/util/workflow/graphToWorkflow';
import type { NonNullableGraph } from 'services/api/types';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

const imageCollectionTemplate = {
  title: 'Image Collection Primitive',
  type: 'image_collection',
  version: '1.0.2',
  tags: ['primitives', 'image', 'collection'],
  description: 'A collection of image primitive values',
  outputType: 'image_collection_output',
  inputs: {
    collection: {
      name: 'collection',
      title: 'Collection',
      required: false,
      description: 'An optional image collection to append to',
      fieldKind: 'input',
      input: 'connection',
      ui_hidden: false,
      type: { name: 'ImageField', cardinality: 'COLLECTION', batch: false },
      default: undefined,
    },
    images: {
      name: 'images',
      title: 'Images',
      required: false,
      description: 'The images to append to the collection',
      fieldKind: 'input',
      input: 'direct',
      ui_hidden: false,
      type: { name: 'ImageField', cardinality: 'COLLECTION', batch: false },
      default: undefined,
    },
  },
  outputs: {
    collection: {
      fieldKind: 'output',
      name: 'collection',
      title: 'Collection',
      description: 'The output images',
      type: { name: 'ImageField', cardinality: 'COLLECTION', batch: false },
      ui_hidden: false,
    },
  },
  useCache: true,
  nodePack: 'invokeai',
  classification: 'stable',
  category: 'primitives',
} satisfies InvocationTemplate;

describe('graphToWorkflow', () => {
  const originalTemplates = $templates.get();

  beforeEach(() => {
    $templates.set({ image_collection: imageCollectionTemplate });
  });

  afterEach(() => {
    $templates.set(originalTemplates);
  });

  it('moves legacy image_collection graph collection values to the visible images field', () => {
    const images = [{ image_name: 'legacy.png' }];
    const graph: NonNullableGraph = {
      id: 'graph',
      nodes: {
        image_collection: {
          id: 'image_collection',
          type: 'image_collection',
          collection: images,
        },
      },
      edges: [],
    };

    const workflow = graphToWorkflow(graph, false);
    const node = workflow.nodes[0];

    if (!node || !isWorkflowInvocationNode(node)) {
      throw new Error('Expected an image_collection workflow node');
    }
    expect(node.data.inputs.images?.value).toEqual(images);
    expect(node.data.inputs.collection?.value).toEqual([]);
  });

  it('preserves legacy image_collection graph collection values when collection is connected', () => {
    const images = [{ image_name: 'shadowed.png' }];
    const graph: NonNullableGraph = {
      id: 'graph',
      nodes: {
        source: {
          id: 'source',
          type: 'image_collection',
        },
        target: {
          id: 'target',
          type: 'image_collection',
          collection: images,
        },
      },
      edges: [
        {
          source: { node_id: 'source', field: 'collection' },
          destination: { node_id: 'target', field: 'collection' },
        },
      ],
    };

    const workflow = graphToWorkflow(graph, false);
    const node = workflow.nodes[1];

    if (!node || !isWorkflowInvocationNode(node)) {
      throw new Error('Expected an image_collection workflow node');
    }
    expect(node.data.inputs.images?.value).toBeUndefined();
    expect(node.data.inputs.collection?.value).toEqual(images);
  });
});
