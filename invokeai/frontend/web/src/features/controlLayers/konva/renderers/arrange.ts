import { BACKGROUND_LAYER_ID, PREVIEW_LAYER_ID } from 'features/controlLayers/konva/naming';
import type { KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';
import type { ControlAdapterEntity, LayerEntity, RegionEntity } from 'features/controlLayers/store/types';
import type Konva from 'konva';

export const arrangeEntities = (
  stage: Konva.Stage,
  layerManager: KonvaNodeManager,
  layers: LayerEntity[],
  controlAdapterManager: KonvaNodeManager,
  controlAdapters: ControlAdapterEntity[],
  regionManager: KonvaNodeManager,
  regions: RegionEntity[]
): void => {
  let zIndex = 0;
  stage.findOne<Konva.Layer>(`#${BACKGROUND_LAYER_ID}`)?.zIndex(++zIndex);
  for (const layer of layers) {
    layerManager.get(layer.id)?.konvaLayer.zIndex(++zIndex);
  }
  for (const ca of controlAdapters) {
    controlAdapterManager.get(ca.id)?.konvaLayer.zIndex(++zIndex);
  }
  for (const rg of regions) {
    regionManager.get(rg.id)?.konvaLayer.zIndex(++zIndex);
  }
  stage.findOne<Konva.Layer>(`#${PREVIEW_LAYER_ID}`)?.zIndex(++zIndex);
};
