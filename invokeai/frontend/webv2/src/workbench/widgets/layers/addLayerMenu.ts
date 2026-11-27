import type { LayerStackKind } from '@workbench/canvas-engine/api';

import { canAddRegionalReferenceImage } from './layerOps';

/** The distinct "add a layer" actions offered across the panel's add surfaces. */
export type AddLayerItemId =
  | 'vector'
  | 'inpaint_mask'
  | 'regional_guidance'
  | 'regional_reference_image'
  | 'control'
  | 'raster'
  | 'group';

/** A single add-layer menu entry (label is an i18n key; the icon comes from `LAYER_KIND_ICONS`). */
export interface AddLayerMenuItem {
  id: AddLayerItemId;
  labelKey: string;
}

export interface AddLayerMenuGroup {
  titleKey: string;
  items: AddLayerMenuItem[];
}

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
      { id: 'vector', labelKey: 'widgets.layers.actions.addVectorLayer' },
      { id: 'group', labelKey: 'widgets.layers.actions.newGroup' },
    ],
    titleKey: 'widgets.layers.menuGroups.layers',
  },
];

export const isAddLayerItemAvailable = (id: AddLayerItemId, base: string | null): boolean =>
  id !== 'regional_reference_image' || canAddRegionalReferenceImage(base);

/** The add-layer action a stack-header "New" button triggers for its type. */
export const stackAddItemId = (stack: LayerStackKind): AddLayerItemId => stack;
