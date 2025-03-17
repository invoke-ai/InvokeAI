import { useAppSelector } from 'app/store/storeHooks';
import { selectIsCogView4, selectIsSD3 } from 'features/controlLayers/store/paramsSlice';
import type { CanvasEntityType } from 'features/controlLayers/store/types';
import { useCallback, useMemo } from 'react';

export const useIsEntityTypeEnabled = () => {
  const isSD3 = useAppSelector(selectIsSD3);
  const isCogView4 = useAppSelector(selectIsCogView4);

  const isEntityTypeEnabled = useCallback(
    (layerType: CanvasEntityType) => {
      switch (layerType) {
        case 'reference_image':
          return !isSD3 && !isCogView4;
        case 'regional_guidance':
          return !isSD3 && !isCogView4;
        case 'control_layer':
          return !isSD3 && !isCogView4;
        case 'inpaint_mask':
          return true;
        case 'raster_layer':
          return true;
        default:
          break;
      }
    },
    [isSD3, isCogView4]
  );

  const isReferenceImageEnabled = useMemo(() => {
    return isEntityTypeEnabled('reference_image');
  }, [isEntityTypeEnabled]);

  const isRegionalGuidanceEnabled = useMemo(() => {
    return isEntityTypeEnabled('regional_guidance');
  }, [isEntityTypeEnabled]);

  const isRasterLayerEnabled = useMemo(() => {
    return isEntityTypeEnabled('raster_layer');
  }, [isEntityTypeEnabled]);

  const isControlLayerEnabled = useMemo(() => {
    return isEntityTypeEnabled('control_layer');
  }, [isEntityTypeEnabled]);

  const isInpaintMaskEnabled = useMemo(() => {
    return isEntityTypeEnabled('inpaint_mask');
  }, [isEntityTypeEnabled]);

  return {
    isReferenceImageEnabled,
    isRegionalGuidanceEnabled,
    isRasterLayerEnabled,
    isControlLayerEnabled,
    isInpaintMaskEnabled,
  };
};
