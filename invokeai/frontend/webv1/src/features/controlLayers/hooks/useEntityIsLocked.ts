import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice, selectEntity } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';

export const useEntityIsLocked = (entityIdentifier: CanvasEntityIdentifier | null) => {
  const selectIsLocked = useMemo(
    () =>
      createSelector(selectCanvasSlice, (canvas) => {
        if (!entityIdentifier) {
          return false;
        }
        const entity = selectEntity(canvas, entityIdentifier);
        if (!entity) {
          return false;
        } else {
          return entity.isLocked;
        }
      }),
    [entityIdentifier]
  );
  const isLocked = useAppSelector(selectIsLocked);
  return isLocked;
};
