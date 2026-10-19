import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectRefImageEntityOrThrow, selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { useMemo } from 'react';

export const useRefImageEntity = (id: string) => {
  const selectEntity = useMemo(
    () =>
      createSelector(selectRefImagesSlice, (refImages) =>
        selectRefImageEntityOrThrow(refImages, id, `useRefImageState(${id})`)
      ),
    [id]
  );
  const entity = useAppSelector(selectEntity);
  return entity;
};
