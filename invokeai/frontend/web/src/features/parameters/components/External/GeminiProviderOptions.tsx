import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { geminiTemperatureChanged, selectGeminiTemperature } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const TEMPERATURE_MARKS = [0, 1, 2];

export const GeminiProviderOptions = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const temperature = useAppSelector(selectGeminiTemperature);

  const onTemperatureChange = useCallback((v: number) => dispatch(geminiTemperatureChanged(v)), [dispatch]);

  return (
    <FormControl>
      <FormLabel>{t('parameters.temperature', 'Temperature')}</FormLabel>
      <CompositeSlider
        value={temperature ?? 1}
        defaultValue={1}
        min={0}
        max={2}
        step={0.1}
        fineStep={0.05}
        onChange={onTemperatureChange}
        marks={TEMPERATURE_MARKS}
      />
      <CompositeNumberInput
        value={temperature ?? 1}
        defaultValue={1}
        min={0}
        max={2}
        step={0.1}
        fineStep={0.05}
        onChange={onTemperatureChange}
      />
    </FormControl>
  );
});

GeminiProviderOptions.displayName = 'GeminiProviderOptions';
