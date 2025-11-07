import { useAppSelector } from 'app/store/storeHooks';
import { selectIsCogView4, selectIsFluxKontext, selectIsSD3 } from 'features/controlLayers/store/paramsSlice';
import type { CanvasEntityType } from 'features/controlLayers/store/types';
import { useMemo } from 'react';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

export const useIsEntityTypeEnabled = (entityType: CanvasEntityType) => {
  const isSD3 = useAppSelector(selectIsSD3);
  const isCogView4 = useAppSelector(selectIsCogView4);
  const isFluxKontext = useAppSelector(selectIsFluxKontext);

  // TODO(psyche): consider using a constant to define which entity types are supported by which model,
  // see invokeai/frontend/web/src/features/modelManagerV2/models.ts for ref
  const isEntityTypeEnabled = useMemo<boolean>(() => {
    switch (entityType) {
      case 'regional_guidance':
        return !isSD3 && !isCogView4 && !isFluxKontext;
      case 'control_layer':
        return !isSD3 && !isCogView4 && !isFluxKontext;
      case 'inpaint_mask':
        return !isFluxKontext;
      case 'raster_layer':
        return !isFluxKontext;
      default:
        assert<Equals<typeof entityType, never>>(false);
    }
  }, [entityType, isSD3, isCogView4, isFluxKontext]);

  return isEntityTypeEnabled;
};
