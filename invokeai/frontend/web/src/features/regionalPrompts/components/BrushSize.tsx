import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { brushSizeChanged } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const BrushSize = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const brushSize = useAppSelector((s) => s.regionalPrompts.brushSize);
  const onChange = useCallback(
    (v: number) => {
      dispatch(brushSizeChanged(v));
    },
    [dispatch]
  );
  return (
    <FormControl orientation="vertical">
      <FormLabel>{t('regionalPrompts.brushSize')}</FormLabel>
      <CompositeSlider min={1} max={100} value={brushSize} onChange={onChange} />
      <CompositeNumberInput min={1} max={500} value={brushSize} onChange={onChange} />
    </FormControl>
  );
};
