/**
 * Pure data for the layers-panel add-layer surfaces (kept free of React so the
 * grouping is unit-testable). The header menu renders {@link ADD_LAYER_MENU}; each
 * group header's "New" button uses {@link groupAddItemId} to add its own type.
 */

import type { LayerGroupKey } from './layerGroups';

/** The distinct "add a layer" actions offered across the panel's add surfaces. */
export type AddLayerItemId = 'inpaint_mask' | 'regional_guidance' | 'regional_reference_image' | 'control' | 'raster';

/** A single add-layer menu entry (label is an i18n key; the icon lives in the view). */
export interface AddLayerMenuItem {
  id: AddLayerItemId;
  labelKey: string;
}

/** A titled group of add-layer menu entries. */
export interface AddLayerMenuGroup {
  titleKey: string;
  items: AddLayerMenuItem[];
}

/**
 * The add-layer menu, split into legacy's two labelled groups: "Regional" (inpaint
 * mask, regional guidance, regional guidance seeded with a reference image) and
 * "Layers" (control, raster).
 */
export const ADD_LAYER_MENU: readonly AddLayerMenuGroup[] = [
  {
    items: [
      { id: 'inpaint_mask', labelKey: 'widgets.layers.actions.addInpaintMask' },
      { id: 'regional_guidance', labelKey: 'widgets.layers.actions.addRegionalGuidance' },
      { id: 'regional_reference_image', labelKey: 'widgets.layers.actions.addRegionalReferenceImage' },
    ],
    titleKey: 'widgets.layers.menuGroups.regional',
  },
  {
    items: [
      { id: 'control', labelKey: 'widgets.layers.actions.addControlLayer' },
      { id: 'raster', labelKey: 'widgets.layers.actions.addRasterLayer' },
    ],
    titleKey: 'widgets.layers.menuGroups.layers',
  },
];

/** Whether an add action is supported by the selected model base. */
export const isAddLayerItemAvailable = (id: AddLayerItemId, base: string | null): boolean =>
  id !== 'regional_reference_image' || base !== 'flux2';

/** The add-layer action a group-header "New" button triggers for its type. */
export const groupAddItemId = (groupKey: LayerGroupKey): AddLayerItemId => groupKey;
