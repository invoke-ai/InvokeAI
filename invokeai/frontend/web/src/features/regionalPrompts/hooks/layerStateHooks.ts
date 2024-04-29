import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import {
  isMaskedGuidanceLayer,
  selectRegionalPromptsSlice,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useLayerPositivePrompt = (layerId: string) => {
  const selectLayer = useMemo(
    () =>
      createSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layer = regionalPrompts.present.layers.find((l) => l.id === layerId);
        assert(isMaskedGuidanceLayer(layer), `Layer ${layerId} not found or not an RP layer`);
        assert(layer.positivePrompt !== null, `Layer ${layerId} does not have a positive prompt`);
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
        assert(isMaskedGuidanceLayer(layer), `Layer ${layerId} not found or not an RP layer`);
        assert(layer.negativePrompt !== null, `Layer ${layerId} does not have a negative prompt`);
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
        assert(layer, `Layer ${layerId} not found`);
        return layer.isEnabled;
      }),
    [layerId]
  );
  const isVisible = useAppSelector(selectLayer);
  return isVisible;
};
