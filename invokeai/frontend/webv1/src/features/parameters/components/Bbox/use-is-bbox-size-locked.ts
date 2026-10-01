import { useAppSelector } from 'app/store/storeHooks';
import { useCanvasIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectHasFixedDimensionSizes } from 'features/controlLayers/store/paramsSlice';

export const useIsBboxSizeLocked = () => {
  const isStaging = useCanvasIsStaging();
  const hasFixedSizes = useAppSelector(selectHasFixedDimensionSizes);
  return isStaging || hasFixedSizes;
};
