/**
 * Regression tests for issue #9151:
 *   "Lora collection not firing during generation on workflow"
 *
 * Actual root cause: the LoRA *is* applied during the second run (verified
 * by the user's generation logs and by inspecting the second-run graph —
 * lora_selector / collect / z_image_lora_collection_loader nodes and edges
 * are identical to the first run). What was broken: `core_metadata.loras`
 * and the sibling metadata lists were dropped during the graph -> workflow
 * -> graph roundtrip, so the resulting image had no LoRA in its metadata
 * and Recall found nothing.
 *
 * Cause: `LoRAMetadataField` (and the ControlNet/IPAdapter/T2IAdapter
 * metadata field types) were not registered as `zStatefulFieldType`, so
 * inputs holding them fell through to `zStatelessFieldInputInstance`,
 * whose `value` is `z.undefined().catch(undefined)` — coercing the value
 * to undefined.
 *
 * These metadata pass-through instances are only declared on nodes that
 * accept extras (`core_metadata`), so they round-trip via the scoped
 * `zFieldInputInstanceWithExtras` union rather than the global one - see
 * PR #9162.
 */

import { zFieldInputInstanceWithExtras } from 'features/nodes/types/field';
import { describe, expect, it } from 'vitest';

describe('issue #9151: metadata-field roundtrip', () => {
  it('preserves a LoRAMetadataField[] value', () => {
    const inputInstance = {
      name: 'loras',
      label: '',
      description: '',
      value: [
        {
          model: {
            key: 'lora-key-1',
            hash: 'hash1',
            name: 'My Z-Image LoRA',
            base: 'z-image',
            type: 'lora',
            submodel_type: null,
          },
          weight: 0.75,
        },
      ],
    };

    const parsed = zFieldInputInstanceWithExtras.parse(inputInstance);
    expect(parsed.value).toEqual(inputInstance.value);
  });

  it('preserves a ControlNetMetadataField[] value', () => {
    const inputInstance = {
      name: 'controlnets',
      label: '',
      description: '',
      value: [
        {
          image: { image_name: 'in.png' },
          control_model: { key: 'cn', hash: 'h', name: 'cn', base: 'sdxl', type: 'controlnet' },
          control_weight: 0.5,
        },
      ],
    };

    expect(zFieldInputInstanceWithExtras.parse(inputInstance).value).toEqual(inputInstance.value);
  });

  it('preserves an IPAdapterMetadataField[] value', () => {
    const inputInstance = {
      name: 'ipAdapters',
      label: '',
      description: '',
      value: [
        {
          image: { image_name: 'in.png' },
          ip_adapter_model: { key: 'ip', hash: 'h', name: 'ip', base: 'sdxl', type: 'ip_adapter' },
          clip_vision_model: 'ViT-H',
          method: 'full',
          weight: 0.6,
          begin_step_percent: 0,
          end_step_percent: 1,
        },
      ],
    };

    expect(zFieldInputInstanceWithExtras.parse(inputInstance).value).toEqual(inputInstance.value);
  });

  it('preserves a T2IAdapterMetadataField[] value', () => {
    const inputInstance = {
      name: 't2iAdapters',
      label: '',
      description: '',
      value: [
        {
          image: { image_name: 'in.png' },
          t2i_adapter_model: { key: 't2i', hash: 'h', name: 't2i', base: 'sdxl', type: 't2i_adapter' },
        },
      ],
    };

    expect(zFieldInputInstanceWithExtras.parse(inputInstance).value).toEqual(inputInstance.value);
  });

  it('null is still accepted (no loras applied case)', () => {
    const inputInstance = {
      name: 'loras',
      label: '',
      description: '',
      value: null,
    };

    expect(zFieldInputInstanceWithExtras.parse(inputInstance).value).toBeNull();
  });
});
