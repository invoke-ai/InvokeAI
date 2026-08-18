import { describe, expect, it } from 'vitest';

import { parseFieldType, parseOpenApiToTemplates } from './templates';

const openApiFixture = {
  components: {
    schemas: {
      AddInvocation: {
        category: 'math',
        class: 'invocation',
        classification: 'stable',
        description: 'Adds two numbers',
        node_pack: 'invokeai',
        output: { $ref: '#/components/schemas/IntegerOutput' },
        properties: {
          a: {
            default: 0,
            field_kind: 'input',
            input: 'any',
            orig_required: false,
            title: 'A',
            type: 'integer',
            ui_hidden: false,
          },
          b: {
            default: 0,
            field_kind: 'input',
            input: 'any',
            minimum: 0,
            orig_required: false,
            title: 'B',
            type: 'integer',
            ui_component: 'video-frame-index',
            ui_hidden: false,
            ui_model_format: ['diffusers'],
          },
          id: { field_kind: 'internal', title: 'Id', type: 'string' },
          is_intermediate: { default: false, field_kind: 'internal', type: 'boolean' },
          type: { const: 'add', default: 'add', title: 'type' },
          use_cache: { default: true, field_kind: 'internal', type: 'boolean' },
        },
        tags: ['math'],
        title: 'Add Integers',
        type: 'object',
        version: '1.0.1',
      },
      DenoiseInvocation: {
        class: 'invocation',
        output: { $ref: '#/components/schemas/LatentsOutput' },
        properties: {
          latents: {
            anyOf: [{ $ref: '#/components/schemas/LatentsField' }, { type: 'null' }],
            field_kind: 'input',
            input: 'connection',
            orig_required: true,
            title: 'Latents',
          },
          prompts: {
            anyOf: [{ items: { type: 'string' }, type: 'array' }, { type: 'string' }],
            field_kind: 'input',
            orig_required: true,
            title: 'Prompts',
          },
          scheduler: {
            default: 'euler',
            enum: ['euler', 'ddim'],
            field_kind: 'input',
            orig_required: false,
            title: 'Scheduler',
            type: 'string',
          },
          type: { const: 'denoise', default: 'denoise', title: 'type' },
          use_cache: { default: true, field_kind: 'internal', type: 'boolean' },
        },
        title: 'DenoiseInvocation',
        type: 'object',
      },
      GraphInvocation: {
        class: 'invocation',
        output: { $ref: '#/components/schemas/IntegerOutput' },
        properties: {
          type: { const: 'graph', default: 'graph', title: 'type' },
        },
        title: 'Graph',
        type: 'object',
      },
      IntegerOutput: {
        class: 'output',
        properties: {
          type: { const: 'integer_output', default: 'integer_output' },
          value: { field_kind: 'output', title: 'Value', type: 'integer' },
        },
        type: 'object',
      },
      SaveVideoInvocation: {
        class: 'invocation',
        output: { $ref: '#/components/schemas/IntegerOutput' },
        properties: {
          board: {
            anyOf: [{ $ref: '#/components/schemas/BoardField' }, { type: 'null' }],
            field_kind: 'internal',
            input: 'direct',
            orig_required: false,
            title: 'Board',
          },
          id: { field_kind: 'node_attribute', title: 'Id', type: 'string' },
          is_intermediate: { default: false, field_kind: 'node_attribute', type: 'boolean' },
          latents: {
            anyOf: [{ $ref: '#/components/schemas/LatentsField' }, { type: 'null' }],
            field_kind: 'input',
            input: 'connection',
            orig_required: true,
            title: 'Latents',
          },
          metadata: {
            anyOf: [{ $ref: '#/components/schemas/MetadataField' }, { type: 'null' }],
            field_kind: 'internal',
            input: 'connection',
            orig_required: false,
            title: 'Metadata',
          },
          type: { const: 'save_video', default: 'save_video', title: 'type' },
          use_cache: { default: true, field_kind: 'node_attribute', type: 'boolean' },
        },
        title: 'Save Video',
        type: 'object',
      },
      LatentsOutput: {
        class: 'output',
        properties: {
          latents: {
            allOf: [{ $ref: '#/components/schemas/LatentsField' }],
            field_kind: 'output',
            title: 'Latents',
          },
          type: { const: 'latents_output', default: 'latents_output' },
        },
        type: 'object',
      },
    },
  },
};

describe('parseOpenApiToTemplates', () => {
  const templates = parseOpenApiToTemplates(openApiFixture);

  it('parses invocation schemas into templates, skipping the denylist', () => {
    expect(Object.keys(templates).sort()).toEqual(['add', 'denoise', 'save_video']);

    const add = templates.add;

    expect(add?.title).toBe('Add Integers');
    expect(add?.version).toBe('1.0.1');
    expect(add?.category).toBe('math');
    expect(add?.outputType).toBe('integer_output');
  });

  it('parses input templates with constraints and skips reserved fields', () => {
    const add = templates.add;

    expect(Object.keys(add?.inputs ?? {}).sort()).toEqual(['a', 'b']);
    expect(add?.inputs.b?.minimum).toBe(0);
    expect(add?.inputs.a?.type).toEqual({ batch: false, cardinality: 'SINGLE', name: 'IntegerField' });
    // Unknown ui_component values collapse to null; known ones pass through.
    expect(add?.inputs.a?.uiComponent).toBeNull();
    expect(add?.inputs.b?.uiComponent).toBe('video-frame-index');
    // ui_model_format passes through so model pickers can filter by install format.
    expect(add?.inputs.a?.uiModelFormat).toBeNull();
    expect(add?.inputs.b?.uiModelFormat).toEqual(['diffusers']);
  });

  it('parses ref, nullable-anyOf, single-or-collection, and enum field types', () => {
    const denoise = templates.denoise;

    expect(denoise?.inputs.latents?.type).toEqual({ batch: false, cardinality: 'SINGLE', name: 'LatentsField' });
    expect(denoise?.inputs.latents?.input).toBe('connection');
    expect(denoise?.inputs.latents?.required).toBe(true);
    expect(denoise?.inputs.prompts?.type).toEqual({
      batch: false,
      cardinality: 'SINGLE_OR_COLLECTION',
      name: 'StringField',
    });
    expect(denoise?.inputs.scheduler?.type.name).toBe('EnumField');
    expect(denoise?.inputs.scheduler?.options).toEqual(['euler', 'ddim']);
  });

  it('keeps internal-kind metadata and board inputs, but drops node attributes', () => {
    // `metadata` and `board` are the only two internal-kind properties in the schema, and both
    // are connectable: without them a bundled workflow's Core Metadata edge has no handle to
    // land on and is dropped on re-save.
    const saveVideo = templates.save_video;

    expect(Object.keys(saveVideo?.inputs ?? {}).sort()).toEqual(['board', 'latents', 'metadata']);
    expect(saveVideo?.inputs.metadata?.type.name).toBe('MetadataField');
    expect(saveVideo?.inputs.board?.type.name).toBe('BoardField');
  });

  it('parses output templates', () => {
    expect(templates.denoise?.outputs.latents?.type.name).toBe('LatentsField');
    expect(templates.add?.outputs.value?.type.name).toBe('IntegerField');
  });
});

describe('parseFieldType', () => {
  it('parses collections of refs and primitives', () => {
    expect(parseFieldType({ items: { $ref: '#/components/schemas/ImageField' }, type: 'array' })).toEqual({
      batch: false,
      cardinality: 'COLLECTION',
      name: 'ImageField',
    });
    expect(parseFieldType({ items: { type: 'integer' }, type: 'array' })).toEqual({
      batch: false,
      cardinality: 'COLLECTION',
      name: 'IntegerField',
    });
  });

  it('returns null for unparseable shapes instead of throwing', () => {
    expect(parseFieldType({ anyOf: [{ type: 'string' }, { type: 'integer' }, { type: 'boolean' }] })).toBeNull();
    expect(parseFieldType('nonsense')).toBeNull();
  });
});
