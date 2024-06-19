import type { EntityToKonvaMap } from 'features/controlLayers/konva/entityToKonvaMap';
import { BACKGROUND_LAYER_ID, PREVIEW_LAYER_ID } from 'features/controlLayers/konva/naming';
import type { ControlAdapterEntity, LayerEntity, RegionEntity } from 'features/controlLayers/store/types';
import type Konva from 'konva';

export const arrangeEntities = (
  stage: Konva.Stage,
  layerMap: EntityToKonvaMap,
  layers: LayerEntity[],
  controlAdapterMap: EntityToKonvaMap,
  controlAdapters: ControlAdapterEntity[],
  regionMap: EntityToKonvaMap,
  regions: RegionEntity[]
): void => {
  let zIndex = 0;
  stage.findOne<Konva.Layer>(`#${BACKGROUND_LAYER_ID}`)?.zIndex(++zIndex);
  for (const layer of layers) {
    layerMap.getMapping(layer.id)?.konvaLayer.zIndex(++zIndex);
  }
  for (const ca of controlAdapters) {
    controlAdapterMap.getMapping(ca.id)?.konvaLayer.zIndex(++zIndex);
  }
  for (const rg of regions) {
    regionMap.getMapping(rg.id)?.konvaLayer.zIndex(++zIndex);
  }
  stage.findOne<Konva.Layer>(`#${PREVIEW_LAYER_ID}`)?.zIndex(++zIndex);
};
