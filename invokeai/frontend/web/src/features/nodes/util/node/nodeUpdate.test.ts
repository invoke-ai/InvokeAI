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

const oldMiniMaxH3DenoiseTemplate = {
  title: 'Denoise - MiniMax H3',
  type: 'minimax_h3_denoise',
  version: '1.2.0',
  tags: ['latents', 'video', 'audio', 'minimax'],
  description: 'Run the MiniMax H3 joint audio-video denoising loop.',
  outputType: 'minimax_h3_denoise_output',
  inputs: {
    num_frames: {
      name: 'num_frames',
      title: 'Number of Frames',
      required: false,
      description: 'Number of output frames at the fixed 24 fps.',
      fieldKind: 'input',
      input: 'any',
      ui_hidden: false,
      type: { name: 'IntegerField', cardinality: 'SINGLE', batch: false },
      default: 124,
    },
  },
  outputs: {},
  useCache: true,
  nodePack: 'invokeai',
  classification: 'prototype',
  category: 'latents',
} satisfies InvocationTemplate;

const currentMiniMaxH3DenoiseTemplate = {
  ...oldMiniMaxH3DenoiseTemplate,
  version: '1.3.0',
  inputs: {
    num_frames: {
      name: 'num_frames',
      title: 'Number of Frames',
      required: false,
      description: 'Only the video VAE 17n+5 grid points are offered.',
      fieldKind: 'input',
      input: 'any',
      ui_hidden: false,
      type: { name: 'EnumField', cardinality: 'SINGLE', batch: false },
      default: '124',
      options: ['5', '90', '107', '124', '141', '345'],
      ui_choice_labels: { '124': '124 frames - 5.17 s' },
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
  it('converts a stored minimax_h3_denoise frame count to the matching choice value', () => {
    const node = buildInvocationNode({ x: 0, y: 0 }, oldMiniMaxH3DenoiseTemplate);
    const numFrames = node.data.inputs.num_frames;
    if (!numFrames) {
      throw new Error('Expected num_frames input');
    }
    numFrames.value = 141;

    const updated = updateNode(node, currentMiniMaxH3DenoiseTemplate, { connectedInputNames: new Set() });

    expect(updated.data.version).toBe('1.3.0');
    // Without the migration the stale number survives the merge and the field renders as nothing.
    expect(updated.data.inputs.num_frames?.value).toBe('141');
  });

  it('falls back to the default when a stored frame count is no longer offered', () => {
    const node = buildInvocationNode({ x: 0, y: 0 }, oldMiniMaxH3DenoiseTemplate);
    const numFrames = node.data.inputs.num_frames;
    if (!numFrames) {
      throw new Error('Expected num_frames input');
    }
    numFrames.value = 22;

    const updated = updateNode(node, currentMiniMaxH3DenoiseTemplate, { connectedInputNames: new Set() });

    expect(updated.data.inputs.num_frames?.value).toBe('124');
  });

  it('leaves an already-migrated choice value alone', () => {
    const node = buildInvocationNode({ x: 0, y: 0 }, currentMiniMaxH3DenoiseTemplate);
    const numFrames = node.data.inputs.num_frames;
    if (!numFrames) {
      throw new Error('Expected num_frames input');
    }
    numFrames.value = '345';

    const updated = updateNode(
      node,
      { ...currentMiniMaxH3DenoiseTemplate, version: '1.4.0' },
      { connectedInputNames: new Set() }
    );

    expect(updated.data.inputs.num_frames?.value).toBe('345');
  });
});
