import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasStagingAreaSessionId } from 'features/controlLayers/store/canvasStagingAreaSlice';

import { useCanvasId } from './useCanvasId';

export const useCanvasSessionId = () => {
  const canvasId = useCanvasId();

  return useAppSelector((state) => selectCanvasStagingAreaSessionId(state, canvasId));
};

export const useScopedCanvasSessionId = (canvasId: string) => {
  return useAppSelector((state) => selectCanvasStagingAreaSessionId(state, canvasId));
};
