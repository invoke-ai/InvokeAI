import { describe, expect, it } from 'vitest';

import { getControlValidationReason } from './controlValidation';

const valid = {
  adapterModel: { base: 'sd-1', type: 'controlnet' },
  controlLoraIndex: 0,
  kind: 'controlnet' as const,
  mainBase: 'sd-1',
  mainVariant: undefined,
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
        controlLoraIndex: 1,
        kind: 'control_lora',
        mainBase: 'flux',
      })
    ).toBe('control_lora_limit');
    expect(
      getControlValidationReason({
        adapterModel: { base: 'flux', type: 'control_lora' },
        controlLoraIndex: 0,
        kind: 'control_lora',
        mainBase: 'flux',
        mainVariant: 'dev_fill',
      })
    ).toBe('flux_fill_control_lora');
  });

  it('accepts only a Z-Image ControlNet model on a Z-Image main model', () => {
    const zImage = {
      adapterModel: { base: 'z-image', type: 'controlnet' },
      controlLoraIndex: 0,
      kind: 'z_image_control' as const,
      mainBase: 'z-image',
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
        zImageControlIndex: 1,
      })
    ).toBe('z_image_control_limit');
  });
});
