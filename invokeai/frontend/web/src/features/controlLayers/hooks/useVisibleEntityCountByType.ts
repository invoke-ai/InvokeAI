import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectActiveControlLayerEntities,
  selectActiveInpaintMaskEntities,
  selectActiveRasterLayerEntities,
  selectActiveReferenceImageEntities,
  selectActiveRegionalGuidanceEntities,
} from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useVisibleEntityCountByType = (type: CanvasEntityIdentifier['type']): number => {
  const selectVisibleEntityCountByType = useMemo(() => {
    switch (type) {
      case 'control_layer':
        return createSelector(selectActiveControlLayerEntities, (entities) => entities.length);
      case 'raster_layer':
        return createSelector(selectActiveRasterLayerEntities, (entities) => entities.length);
      case 'inpaint_mask':
        return createSelector(selectActiveInpaintMaskEntities, (entities) => entities.length);
      case 'regional_guidance':
        return createSelector(selectActiveRegionalGuidanceEntities, (entities) => entities.length);
      case 'reference_image':
        return createSelector(selectActiveReferenceImageEntities, (entities) => entities.length);
      default:
        assert(false, 'Invalid entity type');
    }
  }, [type]);
  const visibleEntityCount = useAppSelector(selectVisibleEntityCountByType);
  return visibleEntityCount;
};
