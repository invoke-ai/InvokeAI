import { useAppSelector } from 'app/store/storeHooks';
import { useCanvasIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectIsApiBaseModel } from 'features/controlLayers/store/paramsSlice';

export const useIsBboxSizeLocked = () => {
  const isStaging = useCanvasIsStaging();
  const isApiModel = useAppSelector(selectIsApiBaseModel);
  return isApiModel || isStaging;
};
