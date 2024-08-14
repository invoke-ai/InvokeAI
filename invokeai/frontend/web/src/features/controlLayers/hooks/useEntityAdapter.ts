import { useStore } from '@nanostores/react';
import { $canvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';

export const useEntityAdapter = (entityIdentifier: CanvasEntityIdentifier) => {
  const canvasManager = useStore($canvasManager);
  console.log(canvasManager);
};
