/**
 * Regression tests for PR #9162 review:
 *   The `MetadataExtraField` catch-all (value: `z.any()`) must only preserve undeclared "extra"
 *   input values for node types that accept extras (pydantic `extra='allow'`, e.g. `core_metadata`).
 *
 *   Workflow inputs are parsed WITHOUT their field template (see `zInvocationNodeData.inputs`), so a
 *   global catch-all would let any stale/malformed connection-only value survive parsing for ANY
 *   node and later leak into the backend graph via `buildNodesGraph`. Scoping the catch-all to
 *   extra-accepting node types prevents that.
 */

import { describe, expect, it } from 'vitest';

import { zInvocationNodeData } from './invocation';

const buildNodeData = (type: string, inputs: Record<string, unknown>) => ({
  id: 'node-1',
  version: '1.0.0',
  nodePack: 'invokeai',
  label: '',
  notes: '',
  type,
  isOpen: true,
  isIntermediate: false,
  useCache: true,
  inputs,
});

// A value that matches no stateful field-instance schema: the `type` discriminator rules out the
// generator/color/model/image schemas, and it is neither a primitive nor a collection. Pre-PR such a
// value coerced to `undefined` via the stateless branch; the `MetadataExtraField` catch-all is what
// preserves it, so it is the ideal probe for the scoping boundary.
const opaqueValue = { type: 'connection-payload', payload: 123 };

describe('zInvocationNodeData: extra-input scoping', () => {
  it('drops a stale value on a NON-extra node (coerced to undefined)', () => {
    const parsed = zInvocationNodeData.parse(
      buildNodeData('some_stateless_node', {
        unet: { name: 'unet', label: '', description: '', value: opaqueValue },
      })
    );
    // The key is preserved (it's a real input on the node) but its value is NOT — it falls through
    // to the stateless branch which coerces to undefined, so it cannot leak into the backend graph.
    expect(parsed.inputs.unet).toBeDefined();
    expect(parsed.inputs.unet?.value).toBeUndefined();
  });

  it('drops a stale array-of-records value on a NON-extra node (would match a metadata passthrough instance)', () => {
    // The metadata pass-through instances (LoRA/ControlNet/...) accept `array(record(string, any))`.
    // On a non-extra node they must NOT be used, so a stale array-of-objects value is coerced away.
    const parsed = zInvocationNodeData.parse(
      buildNodeData('some_stateless_node', {
        metadata: { name: 'metadata', label: '', description: '', value: [{ foo: 'bar' }] },
      })
    );
    expect(parsed.inputs.metadata).toBeDefined();
    expect(parsed.inputs.metadata?.value).toBeUndefined();
  });

  it('preserves a metadata passthrough value (loras) on core_metadata', () => {
    const loras = [{ model: { key: 'k', hash: 'h', name: 'n', base: 'z-image', type: 'lora' }, weight: 0.75 }];
    const parsed = zInvocationNodeData.parse(
      buildNodeData('core_metadata', {
        loras: { name: 'loras', label: '', description: '', value: loras },
      })
    );
    expect(parsed.inputs.loras?.value).toEqual(loras);
  });

  it('preserves an undeclared extra value on core_metadata (extra=allow)', () => {
    const parsed = zInvocationNodeData.parse(
      buildNodeData('core_metadata', {
        some_extra: { name: 'some_extra', label: '', description: '', value: opaqueValue },
      })
    );
    expect(parsed.inputs.some_extra?.value).toEqual(opaqueValue);
  });

  it('preserves primitive extras on core_metadata', () => {
    const parsed = zInvocationNodeData.parse(
      buildNodeData('core_metadata', {
        z_image_seed_variance_enabled: {
          name: 'z_image_seed_variance_enabled',
          label: '',
          description: '',
          value: false,
        },
        z_image_seed_variance_strength: {
          name: 'z_image_seed_variance_strength',
          label: '',
          description: '',
          value: 0.1,
        },
      })
    );
    expect(parsed.inputs.z_image_seed_variance_enabled?.value).toBe(false);
    expect(parsed.inputs.z_image_seed_variance_strength?.value).toBe(0.1);
  });

  it('normalizes webv2 dynamic input templates', () => {
    const parsed = zInvocationNodeData.parse({
      ...buildNodeData('call_saved_workflow', {}),
      dynamicInputTemplates: {
        board: {
          batch: false,
          cardinality: 'SINGLE',
          description: 'Child board',
          fieldKind: 'internal',
          input: 'any',
          name: 'board',
          required: false,
          title: 'Board',
          type: { batch: false, cardinality: 'SINGLE', name: 'BoardField' },
          uiChoiceLabels: null,
          uiComponent: null,
          uiHidden: false,
          uiModelBase: null,
          uiModelFormat: null,
          uiModelType: null,
          uiOrder: null,
        },
      },
    });

    expect(parsed.dynamicInputTemplates.board).toMatchObject({
      fieldKind: 'input',
      name: 'board',
      title: 'Board',
      type: { name: 'BoardField' },
      ui_hidden: false,
    });
  });

  it('accepts legacy-compatible enum dynamic templates from webv2', () => {
    const parsed = zInvocationNodeData.parse({
      ...buildNodeData('call_saved_workflow', {}),
      dynamicInputTemplates: {
        literal: {
          batch: false,
          cardinality: 'SINGLE',
          default: '2',
          description: 'Literal value',
          fieldKind: 'input',
          input: 'any',
          name: 'literal',
          options: ['2'],
          required: false,
          title: 'Literal',
          type: { batch: false, cardinality: 'SINGLE', name: 'EnumField' },
          uiHidden: false,
        },
      },
    });

    expect(parsed.dynamicInputTemplates.literal).toMatchObject({ default: '2', options: ['2'] });
  });

  it('accepts an enum dynamic template whose optional default is unset', () => {
    // webv2 leaves `default` unset for an optional enum whose backend default is
    // null, so the backend receives `None` rather than an invented choice. JSON
    // drops the undefined key, so the template arrives here without it. Legacy's
    // `zEnumFieldInputTemplate.default` is a required `z.string()`, and
    // `zWorkflowV3.parse` throws rather than dropping one template, so the whole
    // workflow fails to open in this editor.
    const parsed = zInvocationNodeData.parse({
      ...buildNodeData('call_saved_workflow', {}),
      dynamicInputTemplates: {
        fidelity: {
          description: 'Input fidelity',
          exclusiveMaximum: null,
          exclusiveMinimum: null,
          fieldKind: 'input',
          input: 'any',
          maximum: null,
          minimum: null,
          multipleOf: null,
          name: 'fidelity',
          options: ['low', 'high'],
          required: false,
          title: 'Input Fidelity',
          type: { batch: false, cardinality: 'SINGLE', name: 'EnumField' },
          uiChoiceLabels: null,
          uiComponent: null,
          uiHidden: false,
          uiModelBase: null,
          uiModelFormat: null,
          uiModelType: null,
          uiOrder: null,
        },
      },
    });

    expect(parsed.dynamicInputTemplates.fidelity).toMatchObject({ default: 'low', options: ['low', 'high'] });
  });

  it('accepts an optional enum dynamic template with no options at the legacy boundary', () => {
    const parsed = zInvocationNodeData.parse({
      ...buildNodeData('call_saved_workflow', {}),
      dynamicInputTemplates: {
        empty: {
          description: 'Empty enum',
          fieldKind: 'input',
          input: 'any',
          name: 'empty',
          options: [],
          required: false,
          title: 'Empty enum',
          type: { batch: false, cardinality: 'SINGLE', name: 'EnumField' },
          uiHidden: false,
        },
      },
    });

    expect(parsed.dynamicInputTemplates.empty).toMatchObject({ default: '', options: [] });
  });
});
