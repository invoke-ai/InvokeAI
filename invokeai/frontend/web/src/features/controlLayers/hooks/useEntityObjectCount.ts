import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasV2Slice, selectEntity } from 'features/controlLayers/store/canvasV2Slice';
import { type CanvasEntityIdentifier, isDrawableEntity } from 'features/controlLayers/store/types';
import { useMemo } from 'react';

export const useEntityObjectCount = (entityIdentifier: CanvasEntityIdentifier) => {
  const selectObjectCount = useMemo(
    () =>
      createSelector(selectCanvasV2Slice, (canvasV2) => {
        const entity = selectEntity(canvasV2, entityIdentifier);
        if (!entity) {
          return 0;
        } else if (isDrawableEntity(entity)) {
          return entity.objects.length;
        } else {
          return 0;
        }
      }),
    [entityIdentifier]
  );
  const objectCount = useAppSelector(selectObjectCount);
  return objectCount;
};
