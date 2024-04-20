import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { isVectorMaskLayer, selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useMaskLayerTextPrompt = (layerId: string) => {
  const selectLayer = useMemo(
    () =>
      createSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layer = regionalPrompts.present.layers.find((l) => l.id === layerId);
        assert(isVectorMaskLayer(layer), `Layer ${layerId} not found or not an RP layer`);
        assert(layer.textPrompt !== null, `Layer ${layerId} does not have a text prompt`);
        return layer.textPrompt;
      }),
    [layerId]
  );
  const textPrompt = useAppSelector(selectLayer);
  return textPrompt;
};

export const useLayerIsVisible = (layerId: string) => {
  const selectLayer = useMemo(
    () =>
      createSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layer = regionalPrompts.present.layers.find((l) => l.id === layerId);
        assert(isVectorMaskLayer(layer), `Layer ${layerId} not found or not an RP layer`);
        return layer.isVisible;
      }),
    [layerId]
  );
  const isVisible = useAppSelector(selectLayer);
  return isVisible;
};
