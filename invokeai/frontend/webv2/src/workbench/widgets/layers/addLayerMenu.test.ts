import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import { isAddLayerItemAvailable, stackAddItemId } from './addLayerMenu';

seedArchitectureCapabilities();

describe('isAddLayerItemAvailable', () => {
  it('hides only regional reference-image creation on bases without a regional image path', () => {
    for (const base of ['flux2', 'krea-2', 'z-image', 'anima', 'sd-3']) {
      expect(isAddLayerItemAvailable('regional_reference_image', base), base).toBe(false);
      expect(isAddLayerItemAvailable('regional_guidance', base), base).toBe(true);
    }
    for (const base of ['sd-1', 'sd-2', 'sdxl', 'flux', null]) {
      expect(isAddLayerItemAvailable('regional_reference_image', base), String(base)).toBe(true);
    }
  });
});

describe('stackAddItemId', () => {
  it('maps a group key to its own add-layer item', () => {
    expect(stackAddItemId('raster')).toBe('raster');
    expect(stackAddItemId('control')).toBe('control');
    expect(stackAddItemId('inpaint_mask')).toBe('inpaint_mask');
    expect(stackAddItemId('regional_guidance')).toBe('regional_guidance');
  });
});
