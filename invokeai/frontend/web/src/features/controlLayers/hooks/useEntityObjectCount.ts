import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice, selectEntity } from 'features/controlLayers/store/selectors';
import { type CanvasEntityIdentifier, isDrawableEntity } from 'features/controlLayers/store/types';
import { useMemo } from 'react';

export const useEntityObjectCount = (entityIdentifier: CanvasEntityIdentifier) => {
  const selectObjectCount = useMemo(
    () =>
      createSelector(selectCanvasSlice, (canvas) => {
        const entity = selectEntity(canvas, entityIdentifier);
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
