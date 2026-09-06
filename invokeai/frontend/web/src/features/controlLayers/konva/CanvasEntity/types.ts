import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterInpaintMask } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterInpaintMask';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasEntityAdapterRegionalGuidance } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRegionalGuidance';
import type { CanvasEntityAdapterVectorLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterVectorLayer';
import type { CanvasEntityType } from 'features/controlLayers/store/types';

export type CanvasEntityAdapter =
  | CanvasEntityAdapterRasterLayer
  | CanvasEntityAdapterControlLayer
  | CanvasEntityAdapterVectorLayer
  | CanvasEntityAdapterInpaintMask
  | CanvasEntityAdapterRegionalGuidance;

export type CanvasEntityAdapterFromType<T extends CanvasEntityType> = Extract<
  CanvasEntityAdapter,
  { state: { type: T } }
>;
