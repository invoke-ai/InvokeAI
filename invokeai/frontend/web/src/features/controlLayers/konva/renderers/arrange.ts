import type { KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';

/**
 * Gets a function to arrange the entities in the konva stage.
 * @param manager The konva node manager
 * @returns An arrange entities function
 */
export const getArrangeEntities = (manager: KonvaNodeManager) => {
  const { getLayersState, getControlAdaptersState, getRegionsState } = manager.stateApi;

  function arrangeEntities(): void {
    const layers = getLayersState().entities;
    const controlAdapters = getControlAdaptersState().entities;
    const regions = getRegionsState().entities;
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
