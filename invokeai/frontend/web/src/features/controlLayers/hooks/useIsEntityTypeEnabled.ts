import { useAppSelector } from 'app/store/storeHooks';
import {
  selectIsChatGPT4o,
  selectIsCogView4,
  selectIsFluxKontext,
  selectIsGemini2_5,
  selectIsImagen3,
  selectIsImagen4,
  selectIsSD3,
} from 'features/controlLayers/store/paramsSlice';
import type { CanvasEntityType } from 'features/controlLayers/store/types';
import { useMemo } from 'react';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

export const useIsEntityTypeEnabled = (entityType: CanvasEntityType) => {
  const isSD3 = useAppSelector(selectIsSD3);
  const isCogView4 = useAppSelector(selectIsCogView4);
  const isImagen3 = useAppSelector(selectIsImagen3);
  const isImagen4 = useAppSelector(selectIsImagen4);
  const isFluxKontext = useAppSelector(selectIsFluxKontext);
  const isChatGPT4o = useAppSelector(selectIsChatGPT4o);
  const isGemini2_5 = useAppSelector(selectIsGemini2_5);

  const isEntityTypeEnabled = useMemo<boolean>(() => {
    switch (entityType) {
      case 'regional_guidance':
        return !isSD3 && !isCogView4 && !isImagen3 && !isImagen4 && !isFluxKontext && !isChatGPT4o && !isGemini2_5;
      case 'control_layer':
        return !isSD3 && !isCogView4 && !isImagen3 && !isImagen4 && !isFluxKontext && !isChatGPT4o && !isGemini2_5;
      case 'inpaint_mask':
        return !isImagen3 && !isImagen4 && !isFluxKontext && !isChatGPT4o && !isGemini2_5;
      case 'raster_layer':
        return !isImagen3 && !isImagen4 && !isFluxKontext && !isChatGPT4o && !isGemini2_5;
      default:
        assert<Equals<typeof entityType, never>>(false);
    }
  }, [entityType, isSD3, isCogView4, isImagen3, isImagen4, isFluxKontext, isChatGPT4o, isGemini2_5]);

  return isEntityTypeEnabled;
};
