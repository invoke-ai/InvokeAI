import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  geminiTemperatureChanged,
  geminiThinkingLevelChanged,
  selectGeminiTemperature,
  selectGeminiThinkingLevel,
} from 'features/controlLayers/store/paramsSlice';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

const TEMPERATURE_MARKS = [0, 1, 2];

export const GeminiProviderOptions = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const temperature = useAppSelector(selectGeminiTemperature);
  const thinkingLevel = useAppSelector(selectGeminiThinkingLevel);

  const onTemperatureChange = useCallback((v: number) => dispatch(geminiTemperatureChanged(v)), [dispatch]);

  const onThinkingLevelChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => {
      const value = e.target.value;
      dispatch(geminiThinkingLevelChanged(value === '' ? null : (value as 'minimal' | 'high')));
    },
    [dispatch]
  );

  return (
    <>
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
      <FormControl>
        <FormLabel>{t('parameters.thinkingLevel', 'Thinking Level')}</FormLabel>
        <Select
          size="sm"
          value={thinkingLevel ?? ''}
          onChange={onThinkingLevelChange}
          icon={<PiCaretDownBold />}
          iconSize="0.75rem"
        >
          <option value="">Default</option>
          <option value="minimal">Minimal</option>
          <option value="high">High</option>
        </Select>
      </FormControl>
    </>
  );
});

GeminiProviderOptions.displayName = 'GeminiProviderOptions';
