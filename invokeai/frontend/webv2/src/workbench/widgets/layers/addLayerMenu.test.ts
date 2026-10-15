import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import { ADD_LAYER_MENU, isAddLayerItemAvailable, stackAddItemId } from './addLayerMenu';

seedArchitectureCapabilities();

describe('ADD_LAYER_MENU', () => {
  it('splits into the legacy Regional / Layers groups in order', () => {
    expect(ADD_LAYER_MENU.map((group) => group.titleKey)).toEqual([
      'widgets.layers.menuGroups.regional',
      'widgets.layers.menuGroups.layers',
    ]);
  });

  it('lists inpaint mask, regional guidance, and regional reference image under Regional', () => {
    expect(ADD_LAYER_MENU[0]?.items.map((item) => item.id)).toEqual([
      'inpaint_mask',
      'regional_guidance',
      'regional_reference_image',
    ]);
  });

  it('lists control + raster under Layers', () => {
    expect(ADD_LAYER_MENU[1]?.items.map((item) => item.id)).toEqual(['control', 'raster', 'group']);
  });

  it('gives every item a widgets.layers i18n label key', () => {
    for (const group of ADD_LAYER_MENU) {
      for (const item of group.items) {
        expect(item.labelKey.startsWith('widgets.layers.actions.')).toBe(true);
      }
    }
  });
});

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
    // A group header's "New" creates a plain region, not the ref-image variant.
    expect(stackAddItemId('regional_guidance')).toBe('regional_guidance');
  });
});
