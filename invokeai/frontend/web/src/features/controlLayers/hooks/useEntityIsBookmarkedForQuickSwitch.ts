import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectActiveCanvas } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';

export const useEntityIsBookmarkedForQuickSwitch = (entityIdentifier: CanvasEntityIdentifier) => {
  const selectIsBookmarkedForQuickSwitch = useMemo(
    () =>
      createSelector(selectActiveCanvas, (canvas) => {
        return canvas.bookmarkedEntityIdentifier?.id === entityIdentifier.id;
      }),
    [entityIdentifier]
  );
  const isBookmarkedForQuickSwitch = useAppSelector(selectIsBookmarkedForQuickSwitch);

  return isBookmarkedForQuickSwitch;
};
