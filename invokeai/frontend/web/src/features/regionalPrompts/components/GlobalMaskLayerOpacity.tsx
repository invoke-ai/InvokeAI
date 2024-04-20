import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  globalMaskLayerOpacityChanged,
  initialRegionalPromptsState,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const GlobalMaskLayerOpacity = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const globalMaskLayerOpacity = useAppSelector((s) => s.regionalPrompts.present.globalMaskLayerOpacity);
  const onChange = useCallback(
    (v: number) => {
      dispatch(globalMaskLayerOpacityChanged(v));
    },
    [dispatch]
  );
  return (
    <FormControl>
      <FormLabel>{t('regionalPrompts.layerOpacity')}</FormLabel>
      <CompositeSlider
        min={0.25}
        max={1}
        step={0.01}
        value={globalMaskLayerOpacity}
        defaultValue={initialRegionalPromptsState.globalMaskLayerOpacity}
        onChange={onChange}
      />
      <CompositeNumberInput
        min={0.25}
        max={1}
        step={0.01}
        value={globalMaskLayerOpacity}
        defaultValue={initialRegionalPromptsState.globalMaskLayerOpacity}
        onChange={onChange}
      />
    </FormControl>
  );
});

GlobalMaskLayerOpacity.displayName = 'GlobalMaskLayerOpacity';
