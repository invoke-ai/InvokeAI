import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  isDynamicPromptRandomRefreshMode,
  randomRefreshModeChanged,
  selectDynamicPromptsMode,
  selectDynamicPromptsRandomRefreshMode,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamDynamicPromptsRandomRefreshMode = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const mode = useAppSelector(selectDynamicPromptsMode);
  const randomRefreshMode = useAppSelector(selectDynamicPromptsRandomRefreshMode);

  const options = useMemo<ComboboxOption[]>(
    () => [
      {
        value: 'per_image',
        label: t('dynamicPrompts.randomness.perImageLabel'),
        description: t('dynamicPrompts.randomness.perImageDesc'),
      },
      {
        value: 'per_enqueue',
        label: t('dynamicPrompts.randomness.perInvokeLabel'),
        description: t('dynamicPrompts.randomness.perInvokeDesc'),
      },
      {
        value: 'manual',
        label: t('dynamicPrompts.randomness.manualLabel'),
        description: t('dynamicPrompts.randomness.manualDesc'),
      },
    ],
    [t]
  );

  const handleChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isDynamicPromptRandomRefreshMode(v?.value)) {
        return;
      }
      dispatch(randomRefreshModeChanged(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => options.find((o) => o.value === randomRefreshMode), [options, randomRefreshMode]);

  if (mode !== 'random') {
    return null;
  }

  return (
    <FormControl>
      <FormLabel>{t('dynamicPrompts.randomness.label')}</FormLabel>
      <Combobox value={value} options={options} onChange={handleChange} />
    </FormControl>
  );
};

export default memo(ParamDynamicPromptsRandomRefreshMode);
