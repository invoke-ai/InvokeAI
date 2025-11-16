import { useCanvasIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';

export const useIsBboxSizeLocked = () => {
  const isStaging = useCanvasIsStaging();
  return isStaging;
};
