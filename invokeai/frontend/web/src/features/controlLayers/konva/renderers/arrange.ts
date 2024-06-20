import type { KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';
import type { ControlAdapterEntity, LayerEntity, RegionEntity } from 'features/controlLayers/store/types';

export const arrangeEntities = (
  manager: KonvaNodeManager,
  layers: LayerEntity[],
  controlAdapters: ControlAdapterEntity[],
  regions: RegionEntity[]
): void => {
  let zIndex = 0;
  manager.background.layer.zIndex(++zIndex);
  for (const layer of layers) {
    manager.get(layer.id)?.konvaLayer.zIndex(++zIndex);
  }
  for (const ca of controlAdapters) {
    manager.get(ca.id)?.konvaLayer.zIndex(++zIndex);
  }
  for (const rg of regions) {
    manager.get(rg.id)?.konvaLayer.zIndex(++zIndex);
  }
  manager.get('inpaint_mask')?.konvaLayer.zIndex(++zIndex);
  manager.preview.layer.zIndex(++zIndex);
};
