import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { promptLayerOpacityChanged } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const PromptLayerOpacity = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const promptLayerOpacity = useAppSelector((s) => s.regionalPrompts.promptLayerOpacity);
  const onChange = useCallback(
    (v: number) => {
      dispatch(promptLayerOpacityChanged(v));
    },
    [dispatch]
  );
  return (
    <FormControl>
      <FormLabel>Layer Opacity</FormLabel>
      <CompositeSlider min={0.25} max={1} step={0.01} value={promptLayerOpacity} onChange={onChange} />
      <CompositeNumberInput min={0.25} max={1} step={0.01} value={promptLayerOpacity} onChange={onChange} />
    </FormControl>
  );
});

PromptLayerOpacity.displayName = 'PromptLayerOpacity';
