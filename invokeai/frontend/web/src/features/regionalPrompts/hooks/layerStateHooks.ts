import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { isRegionalPromptLayer, selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useLayerPositivePrompt = (layerId: string) => {
  const selectLayer = useMemo(
    () =>
      createSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layer = regionalPrompts.present.layers.find((l) => l.id === layerId);
        assert(isRegionalPromptLayer(layer), `Layer ${layerId} not found or not an RP layer`);
        return layer.positivePrompt;
      }),
    [layerId]
  );
  const prompt = useAppSelector(selectLayer);
  return prompt;
};

export const useLayerNegativePrompt = (layerId: string) => {
  const selectLayer = useMemo(
    () =>
      createSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layer = regionalPrompts.present.layers.find((l) => l.id === layerId);
        assert(isRegionalPromptLayer(layer), `Layer ${layerId} not found or not an RP layer`);
        return layer.negativePrompt;
      }),
    [layerId]
  );
  const prompt = useAppSelector(selectLayer);
  return prompt;
};

export const useLayerIsVisible = (layerId: string) => {
  const selectLayer = useMemo(
    () =>
      createSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layer = regionalPrompts.present.layers.find((l) => l.id === layerId);
        assert(isRegionalPromptLayer(layer), `Layer ${layerId} not found or not an RP layer`);
        return layer.isVisible;
      }),
    [layerId]
  );
  const isVisible = useAppSelector(selectLayer);
  return isVisible;
};
