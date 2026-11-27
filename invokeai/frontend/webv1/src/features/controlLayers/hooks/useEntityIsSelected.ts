import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';

export const useEntityIsSelected = (entityIdentifier: CanvasEntityIdentifier) => {
  const selectIsSelected = useMemo(
    () =>
      createSelector(selectSelectedEntityIdentifier, (selectedEntityIdentifier) => {
        return selectedEntityIdentifier?.id === entityIdentifier.id;
      }),
    [entityIdentifier.id]
  );
  const isSelected = useAppSelector(selectIsSelected);

  return isSelected;
};
