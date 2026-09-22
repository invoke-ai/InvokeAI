/**
 * Verifies that parseSchema produces a stateful LoRAMetadataField template
 * for core_metadata.loras given the real OpenAPI schema shape that the
 * backend emits.
 *
 * Part of issue #9151 investigation.
 */

import type { OpenAPIV3_1 } from 'openapi-types';
import { describe, expect, it } from 'vitest';

import { parseSchema } from './parseSchema';

const minimalSchema = {
  openapi: '3.1.0',
  info: { title: 'test', version: '0.0.0' },
  components: {
    schemas: {
      LoRAMetadataField: {
        title: 'LoRAMetadataField',
        type: 'object',
        properties: {
          model: { $ref: '#/components/schemas/ModelIdentifierField' },
          weight: { type: 'number', title: 'Weight' },
        },
        required: ['model', 'weight'],
      },
      ModelIdentifierField: {
        title: 'ModelIdentifierField',
        type: 'object',
        properties: {
          key: { type: 'string' },
          hash: { type: 'string' },
          name: { type: 'string' },
          base: { type: 'string' },
          type: { type: 'string' },
        },
        required: ['key', 'hash', 'name', 'base', 'type'],
      },
      CoreMetadataOutput: {
        class: 'output',
        title: 'CoreMetadataOutput',
        type: 'object',
        properties: {
          metadata: { type: 'object', title: 'Metadata', field_kind: 'output' },
          type: { type: 'string', const: 'metadata_output', default: 'metadata_output', field_kind: 'node_attribute' },
        },
      },
      CoreMetadataInvocation: {
        class: 'invocation',
        title: 'CoreMetadataInvocation',
        type: 'object',
        category: 'metadata',
        classification: 'internal',
        node_pack: 'invokeai',
        tags: ['metadata'],
        version: '2.1.0',
        output: { $ref: '#/components/schemas/CoreMetadataOutput' },
        properties: {
          id: { type: 'string', title: 'Id', field_kind: 'node_attribute' },
          is_intermediate: {
            type: 'boolean',
            title: 'Is Intermediate',
            default: false,
            field_kind: 'node_attribute',
            ui_hidden: false,
            ui_type: 'IsIntermediate',
            input: 'direct',
            orig_required: true,
          },
          use_cache: { type: 'boolean', default: true, title: 'Use Cache', field_kind: 'node_attribute' },
          type: {
            type: 'string',
            const: 'core_metadata',
            default: 'core_metadata',
            title: 'type',
            field_kind: 'node_attribute',
          },
          loras: {
            anyOf: [{ items: { $ref: '#/components/schemas/LoRAMetadataField' }, type: 'array' }, { type: 'null' }],
            default: null,
            description: 'The LoRAs used for inference',
            title: 'Loras',
            field_kind: 'input',
            input: 'any',
            orig_default: null,
            orig_required: false,
          },
        },
        required: ['type', 'id'],
      },
    },
  },
} as unknown as OpenAPIV3_1.Document;

describe('parseSchema: core_metadata.loras', () => {
  it('produces a LoRAMetadataField template (not stateless)', () => {
    const templates = parseSchema(minimalSchema);
    const coreMetadata = templates.core_metadata;
    expect(coreMetadata).toBeDefined();

    const lorasInput = coreMetadata?.inputs.loras;
    expect(lorasInput).toBeDefined();
    expect(lorasInput?.type.name).toBe('LoRAMetadataField');
    expect(lorasInput?.type.cardinality).toBe('COLLECTION');
    // Stateless inputs only accept 'connection'. If our template is correctly
    // stateful, this should be 'any' (the input mode declared in the schema).
    expect(lorasInput?.input).toBe('any');
  });
});
