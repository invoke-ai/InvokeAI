import { $canvasBaseLayer } from 'features/canvas/store/canvasNanostore';

import { konvaNodeToBlob } from './konvaNodeToBlob';

/**
 * Gets the canvas base layer blob, without bounding box
 */
export const getFullBaseLayerBlob = async () => {
  const canvasBaseLayer = $canvasBaseLayer.get();

  if (!canvasBaseLayer) {
    return;
  }

  const clonedBaseLayer = canvasBaseLayer.clone();

  clonedBaseLayer.scale({ x: 1, y: 1 });

  return konvaNodeToBlob(clonedBaseLayer, clonedBaseLayer.getClientRect());
};
