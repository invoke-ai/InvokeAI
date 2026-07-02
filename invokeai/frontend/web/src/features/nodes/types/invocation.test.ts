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
});
