import { describe, expect, it } from 'vitest';

import { ADD_LAYER_MENU, groupAddItemId, isAddLayerItemAvailable } from './addLayerMenu';

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
    expect(ADD_LAYER_MENU[1]?.items.map((item) => item.id)).toEqual(['control', 'raster']);
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
  it('hides only regional reference-image creation for FLUX.2', () => {
    expect(isAddLayerItemAvailable('regional_reference_image', 'flux2')).toBe(false);
    expect(isAddLayerItemAvailable('regional_guidance', 'flux2')).toBe(true);
    expect(isAddLayerItemAvailable('regional_reference_image', 'flux')).toBe(true);
  });
});

describe('groupAddItemId', () => {
  it('maps a group key to its own add-layer item', () => {
    expect(groupAddItemId('raster')).toBe('raster');
    expect(groupAddItemId('control')).toBe('control');
    expect(groupAddItemId('inpaint_mask')).toBe('inpaint_mask');
    // A group header's "New" creates a plain region, not the ref-image variant.
    expect(groupAddItemId('regional_guidance')).toBe('regional_guidance');
  });
});
