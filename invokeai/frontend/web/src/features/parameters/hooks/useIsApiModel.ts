import { useAppSelector } from 'app/store/storeHooks';
import { selectIsChatGPT4o, selectIsImagen3, selectIsImagen4 } from 'features/controlLayers/store/paramsSlice';

export const useIsApiModel = () => {
  const isImagen3 = useAppSelector(selectIsImagen3);
  const isImagen4 = useAppSelector(selectIsImagen4);
  const isChatGPT4o = useAppSelector(selectIsChatGPT4o);

  return isImagen3 || isImagen4 || isChatGPT4o;
};
