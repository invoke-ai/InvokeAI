import { useCanvasIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { useIsApiModel } from 'features/parameters/hooks/useIsApiModel';

export const useIsBboxSizeLocked = () => {
  const isStaging = useCanvasIsStaging();
  const isApiModel = useIsApiModel();

  return isApiModel || isStaging;
};
