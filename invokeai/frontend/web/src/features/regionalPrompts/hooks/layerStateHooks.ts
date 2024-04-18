import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useLayerPositivePrompt = (layerId: string) => {
  const selectLayer = useMemo(
    () =>
      createSelector(
        selectRegionalPromptsSlice,
        (regionalPrompts) => regionalPrompts.present.layers.find((l) => l.id === layerId)?.positivePrompt
      ),
    [layerId]
  );
  const prompt = useAppSelector(selectLayer);
  assert(prompt !== undefined, `Layer ${layerId} doesn't exist!`);
  return prompt;
};

export const useLayerNegativePrompt = (layerId: string) => {
  const selectLayer = useMemo(
    () =>
      createSelector(
        selectRegionalPromptsSlice,
        (regionalPrompts) => regionalPrompts.present.layers.find((l) => l.id === layerId)?.negativePrompt
      ),
    [layerId]
  );
  const prompt = useAppSelector(selectLayer);
  assert(prompt !== undefined, `Layer ${layerId} doesn't exist!`);
  return prompt;
};

export const useLayerIsVisible = (layerId: string) => {
  const selectLayer = useMemo(
    () =>
      createSelector(
        selectRegionalPromptsSlice,
        (regionalPrompts) => regionalPrompts.present.layers.find((l) => l.id === layerId)?.isVisible
      ),
    [layerId]
  );
  const isVisible = useAppSelector(selectLayer);
  assert(isVisible !== undefined, `Layer ${layerId} doesn't exist!`);
  return isVisible;
};
