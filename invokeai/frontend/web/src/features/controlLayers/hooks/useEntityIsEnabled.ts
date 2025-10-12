import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectEntity, selectSelectedCanvas } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';

export const useEntityIsEnabled = (entityIdentifier: CanvasEntityIdentifier) => {
  const selectIsEnabled = useMemo(
    () =>
      createSelector(selectSelectedCanvas, (canvas) => {
        const entity = selectEntity(canvas, entityIdentifier);
        if (!entity) {
          return false;
        } else {
          return entity.isEnabled;
        }
      }),
    [entityIdentifier]
  );
  const isEnabled = useAppSelector(selectIsEnabled);
  return isEnabled;
};
