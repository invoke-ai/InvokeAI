import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { rgbColorToString } from 'common/util/colorCodeTransformers';
import { selectCanvasV2Slice, selectEntity } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';

export const useEntitySelectionColor = (entityIdentifier: CanvasEntityIdentifier) => {
  const selectSelectionColor = useMemo(
    () =>
      createSelector(selectCanvasV2Slice, (canvasV2) => {
        const entity = selectEntity(canvasV2, entityIdentifier);
        if (!entity) {
          return 'base.400';
        } else if (entity.type === 'inpaint_mask') {
          return rgbColorToString(entity.fill.color);
        } else if (entity.type === 'regional_guidance') {
          return rgbColorToString(entity.fill.color);
        } else {
          return 'base.400';
        }
      }),
    [entityIdentifier]
  );
  const selectionColor = useAppSelector(selectSelectionColor);
  return selectionColor;
};
