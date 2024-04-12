import { CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { promptLayerOpacityChanged } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const PromptLayerOpacity = () => {
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
    <FormControl orientation="vertical">
      <FormLabel>{t('regionalPrompts.brushSize')}</FormLabel>
      <CompositeSlider min={0.25} max={1} step={0.01} value={promptLayerOpacity} onChange={onChange} />
    </FormControl>
  );
};
