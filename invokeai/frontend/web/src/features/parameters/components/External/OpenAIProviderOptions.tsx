import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  openaiBackgroundChanged,
  openaiInputFidelityChanged,
  openaiQualityChanged,
  selectOpenaiBackground,
  selectOpenaiInputFidelity,
  selectOpenaiQuality,
} from 'features/controlLayers/store/paramsSlice';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

export const OpenAIProviderOptions = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const quality = useAppSelector(selectOpenaiQuality);
  const background = useAppSelector(selectOpenaiBackground);
  const inputFidelity = useAppSelector(selectOpenaiInputFidelity);

  const onQualityChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => dispatch(openaiQualityChanged(e.target.value as 'auto' | 'high' | 'medium' | 'low')),
    [dispatch]
  );

  const onBackgroundChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => dispatch(openaiBackgroundChanged(e.target.value as 'auto' | 'transparent' | 'opaque')),
    [dispatch]
  );

  const onInputFidelityChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => {
      const value = e.target.value;
      dispatch(openaiInputFidelityChanged(value === '' ? null : (value as 'low' | 'high')));
    },
    [dispatch]
  );

  return (
    <>
      <FormControl>
        <FormLabel>{t('parameters.quality', 'Quality')}</FormLabel>
        <Select size="sm" value={quality} onChange={onQualityChange} icon={<PiCaretDownBold />} iconSize="0.75rem">
          <option value="auto">Auto</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
        </Select>
      </FormControl>
      <FormControl>
        <FormLabel>{t('parameters.background', 'Background')}</FormLabel>
        <Select
          size="sm"
          value={background}
          onChange={onBackgroundChange}
          icon={<PiCaretDownBold />}
          iconSize="0.75rem"
        >
          <option value="auto">Auto</option>
          <option value="transparent">Transparent</option>
          <option value="opaque">Opaque</option>
        </Select>
      </FormControl>
      <FormControl>
        <FormLabel>{t('parameters.inputFidelity', 'Input Fidelity')}</FormLabel>
        <Select
          size="sm"
          value={inputFidelity ?? ''}
          onChange={onInputFidelityChange}
          icon={<PiCaretDownBold />}
          iconSize="0.75rem"
        >
          <option value="">Default</option>
          <option value="low">Low</option>
          <option value="high">High</option>
        </Select>
      </FormControl>
    </>
  );
});

OpenAIProviderOptions.displayName = 'OpenAIProviderOptions';
