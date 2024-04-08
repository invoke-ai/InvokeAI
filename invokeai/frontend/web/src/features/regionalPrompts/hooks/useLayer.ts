import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useLayer = (layerId: string) => {
  const selectLayer = useMemo(
    () =>
      createSelector(selectRegionalPromptsSlice, (regionalPrompts) =>
        regionalPrompts.layers.find((l) => l.id === layerId)
      ),
    [layerId]
  );
  const layer = useAppSelector(selectLayer);
  assert(layer, `Layer ${layerId} doesn't exist!`);
  return layer;
};
