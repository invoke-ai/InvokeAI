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

const textLLMInputs = {
  prompt: {
    name: 'prompt',
    title: 'Prompt',
    required: false,
    description: 'Input text prompt.',
    fieldKind: 'input',
    input: 'any',
    ui_hidden: false,
    type: { name: 'StringField', cardinality: 'SINGLE', batch: false },
    default: '',
  },
  system_prompt: {
    name: 'system_prompt',
    title: 'System Prompt',
    required: false,
    description: 'System prompt that guides the model behavior.',
    fieldKind: 'input',
    input: 'any',
    ui_hidden: false,
    type: { name: 'StringField', cardinality: 'SINGLE', batch: false },
    default: '',
  },
  text_llm_model: {
    name: 'text_llm_model',
    title: 'Text LLM Model',
    required: true,
    description: 'The text language model to use for text generation.',
    fieldKind: 'input',
    input: 'any',
    ui_hidden: false,
    type: { name: 'ModelIdentifierField', cardinality: 'SINGLE', batch: false },
    default: undefined,
  },
  max_tokens: {
    name: 'max_tokens',
    title: 'Max Tokens',
    required: false,
    description: 'Maximum number of tokens to generate.',
    fieldKind: 'input',
    input: 'any',
    ui_hidden: false,
    type: { name: 'IntegerField', cardinality: 'SINGLE', batch: false },
    default: 300,
  },
} satisfies InvocationTemplate['inputs'];

const textLLMOutput = {
  value: {
    fieldKind: 'output',
    name: 'value',
    title: 'Value',
    description: 'Generated text',
    type: { name: 'StringField', cardinality: 'SINGLE', batch: false },
    ui_hidden: false,
  },
} satisfies InvocationTemplate['outputs'];

const oldTextLLMTemplate = {
  title: 'Text LLM',
  type: 'text_llm',
  version: '1.0.0',
  tags: ['llm', 'text'],
  description: 'Run a text language model.',
  outputType: 'string_output',
  inputs: textLLMInputs,
  outputs: textLLMOutput,
  useCache: true,
  nodePack: 'invokeai',
  classification: 'beta',
  category: 'llm',
} satisfies InvocationTemplate;

const currentTextLLMTemplate = {
  ...oldTextLLMTemplate,
  version: '1.1.0',
  inputs: {
    ...textLLMInputs,
    seed: {
      name: 'seed',
      title: 'Seed',
      required: false,
      description: 'Seed for random number generation.',
      fieldKind: 'input',
      input: 'any',
      ui_hidden: false,
      type: { name: 'IntegerField', cardinality: 'SINGLE', batch: false },
      default: 0,
    },
  },
} satisfies InvocationTemplate;

describe('updateNode', () => {
  it('adds the seeded default when updating a stored text_llm node', () => {
    const node = buildInvocationNode({ x: 0, y: 0 }, oldTextLLMTemplate);
    node.data.inputs.prompt!.value = 'a cat';

    const updated = updateNode(node, currentTextLLMTemplate, { connectedInputNames: new Set() });

    expect(updated.data.version).toBe('1.1.0');
    expect(updated.data.inputs.prompt?.value).toBe('a cat');
    expect(updated.data.inputs.seed?.value).toBe(0);
  });

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
