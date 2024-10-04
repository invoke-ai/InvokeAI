import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterInpaintMask } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterInpaintMask';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasEntityAdapterRegionalGuidance } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRegionalGuidance';

export type CanvasEntityAdapter =
  | CanvasEntityAdapterRasterLayer
  | CanvasEntityAdapterControlLayer
  | CanvasEntityAdapterInpaintMask
  | CanvasEntityAdapterRegionalGuidance;
