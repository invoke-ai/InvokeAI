import { useAppSelector } from 'app/store/storeHooks';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { useIsApiModel } from 'features/parameters/hooks/useIsApiModel';

export const useIsBboxSizeLocked = () => {
  const isStaging = useAppSelector(selectIsStaging);
  const isApiModel = useIsApiModel();

  return isApiModel || isStaging;
};
