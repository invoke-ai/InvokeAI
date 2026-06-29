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
            ui_hidden: false,
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
    expect(Object.keys(templates).sort()).toEqual(['add', 'denoise']);

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
