/**
 * Share React-free add-layer grouping/gating; headers use {@link ADD_LAYER_MENU}, stack buttons use {@link
 * stackAddItemId}.
 */

import type { LayerStackKind } from '@workbench/canvas-engine/api';

import { canAddRegionalReferenceImage } from './layerOps';

/** The distinct "add a layer" actions offered across the panel's add surfaces. */
export type AddLayerItemId =
  | 'inpaint_mask'
  | 'regional_guidance'
  | 'regional_reference_image'
  | 'control'
  | 'raster'
  | 'group';

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
      { id: 'group', labelKey: 'widgets.layers.actions.newGroup' },
    ],
    titleKey: 'widgets.layers.menuGroups.layers',
  },
];

/** Whether an add action is supported by the selected model base. */
export const isAddLayerItemAvailable = (id: AddLayerItemId, base: string | null): boolean =>
  id !== 'regional_reference_image' || canAddRegionalReferenceImage(base);

/** The add-layer action a stack-header "New" button triggers for its type. */
export const stackAddItemId = (stack: LayerStackKind): AddLayerItemId => stack;
