import { useAppSelector } from 'app/store/storeHooks';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectIsChatGTP4o, selectIsImagen3 } from 'features/controlLayers/store/paramsSlice';

export const useIsBboxSizeLocked = () => {
  const isStaging = useAppSelector(selectIsStaging);
  const isImagen3 = useAppSelector(selectIsImagen3);
  const isChatGPT4o = useAppSelector(selectIsChatGTP4o);
  return isImagen3 || isChatGPT4o || isStaging;
};
