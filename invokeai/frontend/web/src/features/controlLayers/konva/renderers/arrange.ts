import type { KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';

/**
 * Gets a function to arrange the entities in the konva stage.
 * @param manager The konva node manager
 * @returns An arrange entities function
 */
export const getArrangeEntities = (manager: KonvaNodeManager) => {
  const { getLayerEntityStates, getControlAdapterEntityStates, getRegionEntityStates } = manager.stateApi;

  function arrangeEntities(): void {
    const layers = getLayerEntityStates();
    const controlAdapters = getControlAdapterEntityStates();
    const regions = getRegionEntityStates();
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
  }

  return arrangeEntities;
};
