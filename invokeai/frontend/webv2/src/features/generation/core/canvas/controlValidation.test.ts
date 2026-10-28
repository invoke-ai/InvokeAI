import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import {
  createControlValidationSequence,
  getControlValidationReason,
  getSuggestedControlKind,
  isAnimaInpaintAdapterModel,
  isControlModelUsableForKind,
} from './controlValidation';

const valid = {
  adapterModel: { base: 'sd-1', type: 'controlnet' },
  controlLoraIndex: 0,
  kind: 'controlnet' as const,
  mainBase: 'sd-1',
  mainVariant: undefined,
  beginEndStepPct: [0, 1] as [number, number],
  weight: 0.75,
};

seedArchitectureCapabilities();

describe('getControlValidationReason', () => {
  it('returns stable reason codes for the validation matrix', () => {
    expect(getControlValidationReason(valid)).toBeNull();
    expect(getControlValidationReason({ ...valid, adapterModel: null })).toBe('missing_model');
    expect(getControlValidationReason({ ...valid, mainBase: 'sd-2' })).toBe('unsupported_adapter');
    expect(getControlValidationReason({ ...valid, adapterModel: { base: 'sdxl', type: 'controlnet' } })).toBe(
      'incompatible_base'
    );
    expect(getControlValidationReason({ ...valid, adapterModel: { base: 'sd-1', type: 't2i_adapter' } })).toBe(
      'incompatible_adapter'
    );
    expect(
      getControlValidationReason({
        adapterModel: { base: 'flux', type: 'control_lora' },
        beginEndStepPct: [0, 1],
        controlLoraIndex: 1,
        kind: 'control_lora',
        mainBase: 'flux',
        weight: 0.75,
      })
    ).toBe('control_lora_limit');
    expect(
      getControlValidationReason({
        adapterModel: { base: 'flux', type: 'control_lora' },
        beginEndStepPct: [0, 1],
        controlLoraIndex: 0,
        kind: 'control_lora',
        mainBase: 'flux',
        mainVariant: 'dev_fill',
        weight: 0.75,
      })
    ).toBe('flux_fill_control_lora');
  });

  it('accepts only a Z-Image ControlNet model on a Z-Image main model', () => {
    const zImage = {
      adapterModel: { base: 'z-image', type: 'controlnet' },
      beginEndStepPct: [0, 1] as [number, number],
      controlLoraIndex: 0,
      kind: 'z_image_control' as const,
      mainBase: 'z-image',
      weight: 0.75,
    };
    expect(getControlValidationReason(zImage)).toBeNull();
    expect(getControlValidationReason({ ...zImage, mainBase: 'sd-1' })).toBe('switch_adapter_kind');
    expect(getControlValidationReason({ ...zImage, adapterModel: { base: 'sdxl', type: 'controlnet' } })).toBe(
      'incompatible_base'
    );
    expect(getControlValidationReason({ ...zImage, adapterModel: { base: 'z-image', type: 't2i_adapter' } })).toBe(
      'incompatible_adapter'
    );
    expect(
      getControlValidationReason({
        ...valid,
        adapterModel: { base: 'z-image', type: 'controlnet' },
        mainBase: 'z-image',
      })
    ).toBe('switch_adapter_kind');
  });

  it('blocks a second Z-Image control', () => {
    expect(
      getControlValidationReason({
        adapterModel: { base: 'z-image', type: 'controlnet' },
        controlLoraIndex: 0,
        kind: 'z_image_control',
        mainBase: 'z-image',
        beginEndStepPct: [0, 1],
        weight: 0.75,
        zImageControlIndex: 1,
      })
    ).toBe('z_image_control_limit');
  });

  it('accepts only a Qwen-Image ControlNet model on a Qwen-Image generation model', () => {
    const qwen = {
      adapterModel: { base: 'qwen-image', type: 'controlnet' },
      beginEndStepPct: [0, 1] as [number, number],
      controlLoraIndex: 0,
      kind: 'qwen_image_control' as const,
      mainBase: 'qwen-image',
      weight: 0.85,
    };
    expect(getControlValidationReason(qwen)).toBeNull();
    expect(getControlValidationReason({ ...qwen, mainVariant: 'edit' })).toBe('qwen_image_edit_control');
    expect(getControlValidationReason({ ...qwen, qwenImageControlIndex: 1 })).toBe('qwen_image_control_limit');
    expect(getControlValidationReason({ ...qwen, mainBase: 'z-image' })).toBe('switch_adapter_kind');
    expect(getControlValidationReason({ ...qwen, kind: 'controlnet' })).toBe('switch_adapter_kind');
    expect(getControlValidationReason({ ...qwen, adapterModel: { base: 'z-image', type: 'controlnet' } })).toBe(
      'incompatible_base'
    );
    expect(getControlValidationReason({ ...qwen, adapterModel: { base: 'qwen-image', type: 'lora' } })).toBe(
      'incompatible_adapter'
    );
    // The node's context scale is 0..2; the legacy -1 floor does not apply.
    expect(getControlValidationReason({ ...qwen, weight: 0 })).toBeNull();
    expect(getControlValidationReason({ ...qwen, weight: 2 })).toBeNull();
    expect(getControlValidationReason({ ...qwen, weight: -0.01 })).toBe('invalid_adapter_values');
    expect(getControlValidationReason({ ...qwen, weight: 2.01 })).toBe('invalid_adapter_values');
  });

  it.each([
    { beginEndStepPct: [0, 1] as [number, number], kind: 'controlnet' as const, weight: Number.NaN },
    { beginEndStepPct: [0, 1] as [number, number], kind: 'controlnet' as const, weight: Number.POSITIVE_INFINITY },
    { beginEndStepPct: [0, 1] as [number, number], kind: 'controlnet' as const, weight: -1.01 },
    { beginEndStepPct: [0, 1] as [number, number], kind: 'controlnet' as const, weight: 2.01 },
    { beginEndStepPct: [0, 1] as [number, number], kind: 'z_image_control' as const, weight: -0.01 },
    { beginEndStepPct: [Number.NaN, 1] as [number, number], kind: 'controlnet' as const, weight: 0.75 },
    { beginEndStepPct: [0, Number.POSITIVE_INFINITY] as [number, number], kind: 'controlnet' as const, weight: 0.75 },
    { beginEndStepPct: [-0.01, 1] as [number, number], kind: 'controlnet' as const, weight: 0.75 },
    { beginEndStepPct: [0, 1.01] as [number, number], kind: 'controlnet' as const, weight: 0.75 },
    { beginEndStepPct: [0.5, 0.5] as [number, number], kind: 'controlnet' as const, weight: 0.75 },
    { beginEndStepPct: [0.8, 0.2] as [number, number], kind: 'controlnet' as const, weight: 0.75 },
  ])('rejects malformed adapter values %#', ({ beginEndStepPct, kind, weight }) => {
    const zImage = kind === 'z_image_control';
    expect(
      getControlValidationReason({
        ...valid,
        adapterModel: { base: zImage ? 'z-image' : 'sd-1', type: 'controlnet' },
        beginEndStepPct,
        kind,
        mainBase: zImage ? 'z-image' : 'sd-1',
        weight,
      })
    ).toBe('invalid_adapter_values');
  });

  it('accepts only a FLUX.2 ControlNet model, and only on FLUX.2 [dev]', () => {
    const dev = {
      adapterModel: { base: 'flux2', type: 'controlnet' },
      beginEndStepPct: [0, 1] as [number, number],
      controlLoraIndex: 0,
      kind: 'flux2_control' as const,
      mainBase: 'flux2',
      mainVariant: 'dev',
      weight: 0.75,
    };
    expect(getControlValidationReason(dev)).toBeNull();
    expect(getControlValidationReason({ ...dev, flux2ControlIndex: 1 })).toBe('flux2_control_limit');
    expect(getControlValidationReason({ ...dev, mainBase: 'qwen-image' })).toBe('switch_adapter_kind');
    expect(getControlValidationReason({ ...dev, kind: 'controlnet' })).toBe('switch_adapter_kind');
    expect(getControlValidationReason({ ...dev, adapterModel: { base: 'flux', type: 'controlnet' } })).toBe(
      'incompatible_base'
    );
    expect(getControlValidationReason({ ...dev, adapterModel: { base: 'flux2', type: 'lora' } })).toBe(
      'incompatible_adapter'
    );
    expect(getControlValidationReason({ ...dev, weight: 0 })).toBeNull();
    expect(getControlValidationReason({ ...dev, weight: 2 })).toBeNull();
    expect(getControlValidationReason({ ...dev, weight: -0.01 })).toBe('invalid_adapter_values');
    expect(getControlValidationReason({ ...dev, weight: 2.01 })).toBe('invalid_adapter_values');
  });

  it.each(['klein_4b', 'klein_9b', 'klein_9b_base', undefined])(
    'refuses the FLUX.2 control on variant %s before asking for a model or valid values',
    (mainVariant) => {
      // Capabilities are per base, so the variant is the policy's to judge; the denoise refuses everything but dev.
      const klein = {
        adapterModel: { base: 'flux2', type: 'controlnet' },
        beginEndStepPct: [0, 1] as [number, number],
        controlLoraIndex: 0,
        kind: 'flux2_control' as const,
        mainBase: 'flux2',
        mainVariant,
        weight: 0.75,
      };
      expect(getControlValidationReason(klein)).toBe('flux2_klein_control');
      expect(getControlValidationReason({ ...klein, adapterModel: null })).toBe('flux2_klein_control');
      expect(getControlValidationReason({ ...klein, weight: -1 })).toBe('flux2_klein_control');
    }
  );

  it('retains the legacy -1 lower weight bound for existing adapter kinds', () => {
    expect(getControlValidationReason({ ...valid, weight: -1 })).toBeNull();
  });
});

describe('Anima ControlNet-LLLite validation', () => {
  const lllite = {
    adapterModel: { base: 'anima', cond_in_channels: 3, type: 'controlnet' },
    beginEndStepPct: [0, 1] as [number, number],
    controlLoraIndex: 0,
    kind: 'anima_lllite' as const,
    mainBase: 'anima',
    weight: 1,
  };

  it('accepts a 3-channel Anima LLLite model on an Anima main model, including the Qwen3.5 variant', () => {
    expect(getControlValidationReason(lllite)).toBeNull();
    expect(getControlValidationReason({ ...lllite, mainVariant: 'anima_qwen35' })).toBeNull();
  });

  it('rejects incompatible pairings with the reason that names the problem', () => {
    expect(getControlValidationReason({ ...lllite, mainBase: 'sdxl' })).toBe('switch_adapter_kind');
    expect(getControlValidationReason({ ...lllite, adapterModel: { ...lllite.adapterModel, base: 'sdxl' } })).toBe(
      'incompatible_base'
    );
    expect(
      getControlValidationReason({ ...lllite, adapterModel: { ...lllite.adapterModel, type: 't2i_adapter' } })
    ).toBe('incompatible_adapter');
    expect(
      getControlValidationReason({ ...lllite, adapterModel: { ...lllite.adapterModel, cond_in_channels: 4 } })
    ).toBe('lllite_inpaint_adapter');
    expect(
      getControlValidationReason({ ...lllite, adapterModel: { ...lllite.adapterModel, cond_in_channels: null } })
    ).toBe('lllite_channels_unknown');
    expect(getControlValidationReason({ ...lllite, adapterModel: { base: 'anima', type: 'controlnet' } })).toBe(
      'lllite_channels_unknown'
    );
    expect(getControlValidationReason({ ...lllite, lliteModelInUse: true })).toBe('duplicate_lllite_model');
  });

  it('asks an Anima layer persisted as ControlNet to switch kind, before asking for a model or valid values', () => {
    expect(getControlValidationReason({ ...lllite, kind: 'controlnet' })).toBe('switch_adapter_kind');
    expect(getControlValidationReason({ ...lllite, adapterModel: null, kind: 'controlnet' })).toBe(
      'switch_adapter_kind'
    );
    expect(getControlValidationReason({ ...lllite, kind: 'controlnet', weight: 5 })).toBe('switch_adapter_kind');
  });

  it('suggests the kind each base runs, preferring ControlNet, and none where control layers are unsupported', () => {
    expect(getSuggestedControlKind('anima')).toBe('anima_lllite');
    expect(getSuggestedControlKind('z-image')).toBe('z_image_control');
    expect(getSuggestedControlKind('qwen-image')).toBe('qwen_image_control');
    expect(getSuggestedControlKind('flux2')).toBe('flux2_control');
    expect(getSuggestedControlKind('flux')).toBe('controlnet');
    expect(getSuggestedControlKind('sd-2')).toBeNull();
    expect(getControlValidationReason({ ...lllite, mainBase: 'sd-2' })).toBe('unsupported_adapter');
  });

  it('holds LLLite weights to the shared -1..2 range, inside the node bounds of -10..10', () => {
    expect(getControlValidationReason({ ...lllite, weight: -1 })).toBeNull();
    expect(getControlValidationReason({ ...lllite, weight: 2 })).toBeNull();
    expect(getControlValidationReason({ ...lllite, weight: 2.01 })).toBe('invalid_adapter_values');
    expect(getControlValidationReason({ ...lllite, weight: -1.01 })).toBe('invalid_adapter_values');
  });

  it('admits the inpaint adapter slot to 4-channel and pre-field installs, never to control adapters', () => {
    const adapter = { base: 'anima', cond_in_channels: 4, type: 'controlnet' };
    expect(isAnimaInpaintAdapterModel(adapter)).toBe(true);
    expect(isAnimaInpaintAdapterModel({ ...adapter, cond_in_channels: null })).toBe(true);
    expect(isAnimaInpaintAdapterModel({ base: 'anima', type: 'controlnet' })).toBe(true);
    expect(isAnimaInpaintAdapterModel({ ...adapter, cond_in_channels: 3 })).toBe(false);
    expect(isAnimaInpaintAdapterModel({ ...adapter, base: 'sdxl' })).toBe(false);
    expect(isAnimaInpaintAdapterModel({ ...adapter, type: 'lora' })).toBe(false);
    // The two slots partition the installed Anima LLLite models.
    expect(isControlModelUsableForKind(adapter, 'anima_lllite')).toBe(false);
    expect(isControlModelUsableForKind({ ...adapter, cond_in_channels: 3 }, 'anima_lllite')).toBe(true);
  });

  it('offers only models a control layer of the kind can run', () => {
    const control = { base: 'anima', cond_in_channels: 3, type: 'controlnet' };
    expect(isControlModelUsableForKind(control, 'anima_lllite')).toBe(true);
    expect(isControlModelUsableForKind({ ...control, cond_in_channels: 4 }, 'anima_lllite')).toBe(false);
    expect(isControlModelUsableForKind({ ...control, cond_in_channels: null }, 'anima_lllite')).toBe(false);
    expect(isControlModelUsableForKind({ ...control, type: 'lora' }, 'anima_lllite')).toBe(false);
    // Channel count is LLLite policy only.
    expect(isControlModelUsableForKind({ base: 'sdxl', type: 'controlnet' }, 'controlnet')).toBe(true);
  });
});

describe('unresolved control models', () => {
  it('reports why resolution found no model in place of missing_model, after kind and value checks', () => {
    expect(getControlValidationReason({ ...valid, adapterModel: null, unresolvedModel: 'choose_model' })).toBe(
      'choose_model'
    );
    expect(getControlValidationReason({ ...valid, adapterModel: null, unresolvedModel: null })).toBe('missing_model');
    expect(
      getControlValidationReason({ ...valid, adapterModel: null, unresolvedModel: 'no_model_for_type', weight: 5 })
    ).toBe('invalid_adapter_values');
    expect(getControlValidationReason({ ...valid, unresolvedModel: 'no_control_models' })).toBeNull();
  });

  it('carries the reason through the sequence without claiming a slot', () => {
    const validate = createControlValidationSequence({ base: 'flux' });
    const controlLora = {
      adapterModel: { base: 'flux', key: 'lora', type: 'control_lora' },
      beginEndStepPct: [0, 1] as [number, number],
      kind: 'control_lora' as const,
      weight: 1,
    };
    expect(validate({ ...controlLora, adapterModel: null, unresolvedModel: 'missing_type' })).toBe('missing_type');
    expect(validate(controlLora)).toBeNull();
  });
});

describe('createControlValidationSequence', () => {
  const anima = (key: string, cond_in_channels: number | null = 3) => ({
    adapterModel: { base: 'anima', cond_in_channels, key, type: 'controlnet' },
    beginEndStepPct: [0, 1] as [number, number],
    kind: 'anima_lllite' as const,
    weight: 1,
  });

  it('rejects a repeated LLLite model but not distinct ones', () => {
    const validate = createControlValidationSequence({ base: 'anima' });
    expect(validate(anima('sketch'))).toBeNull();
    expect(validate(anima('depth'))).toBeNull();
    expect(validate(anima('sketch'))).toBe('duplicate_lllite_model');
  });

  it('does not let an invalid layer claim a model or a limit slot', () => {
    const validate = createControlValidationSequence({ base: 'anima' });
    expect(validate({ ...anima('sketch'), weight: 5 })).toBe('invalid_adapter_values');
    expect(validate(anima('sketch'))).toBeNull();

    const flux = createControlValidationSequence({ base: 'flux' });
    const controlLora = {
      adapterModel: { base: 'flux', key: 'lora', type: 'control_lora' },
      beginEndStepPct: [0, 1] as [number, number],
      kind: 'control_lora' as const,
      weight: 1,
    };
    expect(flux({ ...controlLora, adapterModel: { ...controlLora.adapterModel, base: 'sdxl' } })).toBe(
      'incompatible_base'
    );
    expect(flux(controlLora)).toBeNull();
    expect(flux(controlLora)).toBe('control_lora_limit');
  });

  it('carries the main variant into the policy', () => {
    const validate = createControlValidationSequence({ base: 'flux', variant: 'dev_fill' });
    expect(
      validate({
        adapterModel: { base: 'flux', key: 'lora', type: 'control_lora' },
        beginEndStepPct: [0, 1],
        kind: 'control_lora',
        weight: 1,
      })
    ).toBe('flux_fill_control_lora');
  });

  it('lets one Qwen-Image control through per generation, and none on an Edit model', () => {
    const qwen = {
      adapterModel: { base: 'qwen-image', key: 'union', type: 'controlnet' },
      beginEndStepPct: [0, 1] as [number, number],
      kind: 'qwen_image_control' as const,
      weight: 0.85,
    };
    const generate = createControlValidationSequence({ base: 'qwen-image' });
    expect(generate(qwen)).toBeNull();
    expect(generate(qwen)).toBe('qwen_image_control_limit');

    // A refused layer claims no slot, so every layer on an Edit model reports the same reason.
    const edit = createControlValidationSequence({ base: 'qwen-image', variant: 'edit' });
    expect(edit(qwen)).toBe('qwen_image_edit_control');
    expect(edit(qwen)).toBe('qwen_image_edit_control');
  });

  it('lets one FLUX.2 control through per generation, and none on Klein', () => {
    const flux2 = {
      adapterModel: { base: 'flux2', key: 'union', type: 'controlnet' },
      beginEndStepPct: [0, 1] as [number, number],
      kind: 'flux2_control' as const,
      weight: 0.75,
    };
    const generate = createControlValidationSequence({ base: 'flux2', variant: 'dev' });
    expect(generate(flux2)).toBeNull();
    expect(generate(flux2)).toBe('flux2_control_limit');

    // A refused layer claims no slot, so every layer on a Klein model reports the same reason.
    const klein = createControlValidationSequence({ base: 'flux2', variant: 'klein_9b' });
    expect(klein(flux2)).toBe('flux2_klein_control');
    expect(klein(flux2)).toBe('flux2_klein_control');
  });
});
