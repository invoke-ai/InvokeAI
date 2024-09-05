import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntityAdapter/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterInpaintMask } from 'features/controlLayers/konva/CanvasEntityAdapter/CanvasEntityAdapterInpaintMask';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntityAdapter/CanvasEntityAdapterRasterLayer';
import type { CanvasEntityAdapterRegionalGuidance } from 'features/controlLayers/konva/CanvasEntityAdapter/CanvasEntityAdapterRegionalGuidance';

export type CanvasEntityAdapter =
  | CanvasEntityAdapterRasterLayer
  | CanvasEntityAdapterControlLayer
  | CanvasEntityAdapterInpaintMask
  | CanvasEntityAdapterRegionalGuidance;
