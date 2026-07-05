import type { InvocationTemplate } from 'features/nodes/types/invocation';
import { buildInvocationNode } from 'features/nodes/util/node/buildInvocationNode';
import { updateNode } from 'features/nodes/util/node/nodeUpdate';
import { describe, expect, it } from 'vitest';

const imageCollectionOutput = {
  collection: {
    fieldKind: 'output',
    name: 'collection',
    title: 'Collection',
    description: 'The output images',
    type: {
      name: 'ImageField',
      cardinality: 'COLLECTION',
      batch: false,
    },
    ui_hidden: false,
  },
} satisfies InvocationTemplate['outputs'];

const oldImageCollectionTemplate = {
  title: 'Image Collection Primitive',
  type: 'image_collection',
  version: '1.0.1',
  tags: ['primitives', 'image', 'collection'],
  description: 'A collection of image primitive values',
  outputType: 'image_collection_output',
  inputs: {
    collection: {
      name: 'collection',
      title: 'Collection',
      required: false,
      description: 'The collection of image values',
      fieldKind: 'input',
      input: 'any',
      ui_hidden: false,
      type: {
        name: 'ImageField',
        cardinality: 'COLLECTION',
        batch: false,
      },
      default: undefined,
    },
  },
  outputs: imageCollectionOutput,
  useCache: true,
  nodePack: 'invokeai',
  classification: 'stable',
  category: 'primitives',
} satisfies InvocationTemplate;

const oldestImageCollectionTemplate = {
  ...oldImageCollectionTemplate,
  version: '1.0.0',
} satisfies InvocationTemplate;

const currentImageCollectionTemplate = {
  ...oldImageCollectionTemplate,
  version: '1.0.2',
  inputs: {
    collection: {
      name: 'collection',
      title: 'Collection',
      required: false,
      description: 'An optional image collection to append to',
      fieldKind: 'input',
      input: 'connection',
      ui_hidden: false,
      type: {
        name: 'ImageField',
        cardinality: 'COLLECTION',
        batch: false,
      },
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
      type: {
        name: 'ImageField',
        cardinality: 'COLLECTION',
        batch: false,
      },
      default: undefined,
    },
  },
} satisfies InvocationTemplate;

describe('updateNode', () => {
  it('moves old image_collection direct collection values to the new images field', () => {
    const node = buildInvocationNode({ x: 0, y: 0 }, oldImageCollectionTemplate);
    const images = [{ image_name: 'first' }, { image_name: 'second' }];
    const collectionInput = node.data.inputs.collection;
    if (!collectionInput) {
      throw new Error('Expected collection input');
    }
    collectionInput.value = images;

    const updated = updateNode(node, currentImageCollectionTemplate, { connectedInputNames: new Set() });

    expect(updated.data.version).toBe('1.0.2');
    expect(updated.data.inputs.images?.value).toEqual(images);
    expect(updated.data.inputs.collection?.value).toEqual([]);
  });

  it('moves 1.0.0 image_collection direct collection values to the new images field', () => {
    const node = buildInvocationNode({ x: 0, y: 0 }, oldestImageCollectionTemplate);
    const images = [{ image_name: 'first' }];
    const collectionInput = node.data.inputs.collection;
    if (!collectionInput) {
      throw new Error('Expected collection input');
    }
    collectionInput.value = images;

    const updated = updateNode(node, currentImageCollectionTemplate, { connectedInputNames: new Set() });

    expect(updated.data.version).toBe('1.0.2');
    expect(updated.data.inputs.images?.value).toEqual(images);
    expect(updated.data.inputs.collection?.value).toEqual([]);
  });

  it('preserves old image_collection direct collection values when collection is connected', () => {
    const node = buildInvocationNode({ x: 0, y: 0 }, oldImageCollectionTemplate);
    const images = [{ image_name: 'stale' }];
    const collectionInput = node.data.inputs.collection;
    if (!collectionInput) {
      throw new Error('Expected collection input');
    }
    collectionInput.value = images;

    const updated = updateNode(node, currentImageCollectionTemplate, {
      connectedInputNames: new Set(['collection']),
    });

    expect(updated.data.inputs.images?.value).toBeUndefined();
    expect(updated.data.inputs.collection?.value).toEqual(images);
  });
});
