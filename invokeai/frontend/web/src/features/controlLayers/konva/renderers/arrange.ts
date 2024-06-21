import type { KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';
import type { CanvasV2State } from 'features/controlLayers/store/types';

/**
 * Gets a function to arrange the entities in the konva stage.
 * @param manager The konva node manager
 * @param getLayerEntityStates A function to get all layer entity states
 * @param getControlAdapterEntityStates A function to get all control adapter entity states
 * @param getRegionEntityStates A function to get all region entity states
 * @returns An arrange entities function
 */
export const getArrangeEntities =
  (arg: {
    manager: KonvaNodeManager;
    getLayerEntityStates: () => CanvasV2State['layers']['entities'];
    getControlAdapterEntityStates: () => CanvasV2State['controlAdapters']['entities'];
    getRegionEntityStates: () => CanvasV2State['regions']['entities'];
  }) =>
  (): void => {
    const { manager, getLayerEntityStates, getControlAdapterEntityStates, getRegionEntityStates } = arg;
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
  };
