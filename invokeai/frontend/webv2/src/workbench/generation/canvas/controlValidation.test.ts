import { describe, expect, it } from 'vitest';

import { getControlValidationReason } from './controlValidation';

const valid = {
  adapterModel: { base: 'sd-1', type: 'controlnet' },
  controlLoraIndex: 0,
  kind: 'controlnet' as const,
  mainBase: 'sd-1',
  mainVariant: undefined,
  beginEndStepPct: [0, 1] as [number, number],
  weight: 0.75,
};

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
    expect(getControlValidationReason({ ...zImage, mainBase: 'sd-1' })).toBe('unsupported_adapter');
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
    ).toBe('unsupported_adapter');
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

  it('retains the legacy -1 lower weight bound for existing adapter kinds', () => {
    expect(getControlValidationReason({ ...valid, weight: -1 })).toBeNull();
  });
});
