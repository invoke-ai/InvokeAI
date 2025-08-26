import { useAppSelector } from 'app/store/storeHooks';
import {
  selectIsChatGPT4o,
  selectIsFluxKontextApi,
  selectIsGemini2_5,
  selectIsImagen3,
  selectIsImagen4,
} from 'features/controlLayers/store/paramsSlice';

export const useIsApiModel = () => {
  const isImagen3 = useAppSelector(selectIsImagen3);
  const isImagen4 = useAppSelector(selectIsImagen4);
  const isFluxKontextApi = useAppSelector(selectIsFluxKontextApi);
  const isChatGPT4o = useAppSelector(selectIsChatGPT4o);
  const isGemini2_5 = useAppSelector(selectIsGemini2_5);

  return isImagen3 || isImagen4 || isChatGPT4o || isFluxKontextApi || isGemini2_5;
};
