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
  SplineIcon,
  SquareDashedBottomIcon,
  SquareIcon,
  TypeIcon,
} from 'lucide-react';

import type { AddLayerItemId } from './addLayerMenu';

/** One glyph per creatable layer kind, shared by the add-layer and image destination menus. */
export const LAYER_KIND_ICONS: Readonly<Record<AddLayerItemId, LucideIcon>> = {
  control: SlidersHorizontalIcon,
  group: FolderPlusIcon,
  inpaint_mask: SquareDashedBottomIcon,
  raster: BrushIcon,
  vector: SplineIcon,
  regional_guidance: MapPinIcon,
  regional_reference_image: ImagePlusIcon,
};

/** The small identity glyph between a row's preview and its name. */
export const layerTypeIcon = (layer: CanvasLayerContract): LucideIcon => {
  switch (layer.type) {
    case 'vector':
      return LAYER_KIND_ICONS.vector;
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
