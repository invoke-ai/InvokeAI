import { useAppSelector } from 'app/store/storeHooks';
import { selectIsCogView4, selectIsImagen3, selectIsSD3 } from 'features/controlLayers/store/paramsSlice';
import type { CanvasEntityType } from 'features/controlLayers/store/types';
import { useMemo } from 'react';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

export const useIsEntityTypeEnabled = (entityType: CanvasEntityType) => {
  const isSD3 = useAppSelector(selectIsSD3);
  const isCogView4 = useAppSelector(selectIsCogView4);
  const isImagen3 = useAppSelector(selectIsImagen3);

  const isEntityTypeEnabled = useMemo<boolean>(() => {
    switch (entityType) {
      case 'reference_image':
        return !isSD3 && !isCogView4 && !isImagen3;
      case 'regional_guidance':
        return !isSD3 && !isCogView4 && !isImagen3;
      case 'control_layer':
        return !isSD3 && !isCogView4 && !isImagen3;
      case 'inpaint_mask':
        return !isImagen3;
      case 'raster_layer':
        return !isImagen3;
      default:
        assert<Equals<typeof entityType, never>>(false);
    }
  }, [entityType, isSD3, isCogView4, isImagen3]);

  return isEntityTypeEnabled;
};
