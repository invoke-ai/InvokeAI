import type { CanvasLayerContract } from '@workbench/canvas-engine/api';
import type { LucideIcon } from 'lucide-react';

import {
  BlendIcon,
  BrushIcon,
  FolderPlusIcon,
  ImageIcon,
  ImagePlusIcon,
  MapPinIcon,
  SlidersHorizontalIcon,
  SquareDashedBottomIcon,
  SquareIcon,
  TypeIcon,
} from 'lucide-react';

import type { AddLayerItemId } from './addLayerMenu';

/**
 * One glyph per kind of layer a person can create, wherever a menu offers one: the add-layer menus and the image
 * menus' canvas destinations.
 */
export const LAYER_KIND_ICONS: Readonly<Record<AddLayerItemId, LucideIcon>> = {
  control: SlidersHorizontalIcon,
  group: FolderPlusIcon,
  inpaint_mask: SquareDashedBottomIcon,
  raster: BrushIcon,
  regional_guidance: MapPinIcon,
  regional_reference_image: ImagePlusIcon,
};

/** The small identity glyph between a row's preview and its name. */
export const layerTypeIcon = (layer: CanvasLayerContract): LucideIcon => {
  switch (layer.type) {
    case 'raster':
      switch (layer.source.type) {
        case 'image':
          return ImageIcon;
        case 'shape':
          return SquareIcon;
        case 'text':
          return TypeIcon;
        case 'gradient':
          return BlendIcon;
        default:
          return LAYER_KIND_ICONS.raster;
      }
    case 'control':
      return LAYER_KIND_ICONS.control;
    case 'inpaint_mask':
      return LAYER_KIND_ICONS.inpaint_mask;
    case 'regional_guidance':
      return LAYER_KIND_ICONS.regional_guidance;
  }
};
