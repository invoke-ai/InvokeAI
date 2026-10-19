import type { CanvasLayerContract } from '@workbench/canvas-engine/api';
import type { LucideIcon } from 'lucide-react';

import {
  BlendIcon,
  BrushIcon,
  ImageIcon,
  MapPinIcon,
  SlidersHorizontalIcon,
  SquareDashedBottomIcon,
  SquareIcon,
  TypeIcon,
} from 'lucide-react';

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
          return BrushIcon;
      }
    case 'control':
      return SlidersHorizontalIcon;
    case 'inpaint_mask':
      return SquareDashedBottomIcon;
    case 'regional_guidance':
      return MapPinIcon;
  }
};
