import { useAppSelector } from 'app/store/storeHooks';
import { useActiveCanvasIsStaging } from 'features/controlLayers/hooks/useCanvasIsStaging';
import { selectIsApiBaseModel } from 'features/controlLayers/store/paramsSlice';

export const useIsBboxSizeLocked = () => {
  const isStaging = useActiveCanvasIsStaging();
  const isApiModel = useAppSelector(selectIsApiBaseModel);
  return isApiModel || isStaging;
};
