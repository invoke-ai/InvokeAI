import { useAppSelector } from 'app/store/storeHooks';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectIsGPTImage, selectIsImagen3 } from 'features/controlLayers/store/paramsSlice';

export const useIsBboxSizeLocked = () => {
  const isStaging = useAppSelector(selectIsStaging);
  const isImagen3 = useAppSelector(selectIsImagen3);
  const isGPTImage = useAppSelector(selectIsGPTImage);
  return isImagen3 || isGPTImage || isStaging;
};
