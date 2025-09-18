import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSessionId } from 'features/controlLayers/store/canvasStagingAreaSlice';

import { useCanvasId } from './useCanvasId';

export const useCanvasSessionId = () => {
  const canvasId = useCanvasId();

  return useAppSelector((state) => selectCanvasSessionId(state, canvasId));
};

export const useScopedCanvasSessionId = (canvasId: string) => {
  return useAppSelector((state) => selectCanvasSessionId(state, canvasId));
};
