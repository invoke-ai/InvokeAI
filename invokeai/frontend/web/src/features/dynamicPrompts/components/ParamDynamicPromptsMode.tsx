import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  isDynamicPromptMode,
  modeChanged,
  selectDynamicPromptsMode,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamDynamicPromptsMode = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const mode = useAppSelector(selectDynamicPromptsMode);

  const options = useMemo<ComboboxOption[]>(
    () => [
      {
        value: 'random',
        label: t('dynamicPrompts.mode.randomLabel'),
        description: t('dynamicPrompts.mode.randomDesc'),
      },
      {
        value: 'combinatorial',
        label: t('dynamicPrompts.mode.combinatorialLabel'),
        description: t('dynamicPrompts.mode.combinatorialDesc'),
      },
    ],
    [t]
  );

  const handleChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isDynamicPromptMode(v?.value)) {
        return;
      }
      dispatch(modeChanged(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => options.find((o) => o.value === mode), [mode, options]);

  return (
    <FormControl>
      <FormLabel>{t('dynamicPrompts.mode.label')}</FormLabel>
      <Combobox value={value} options={options} onChange={handleChange} />
    </FormControl>
  );
};

export default memo(ParamDynamicPromptsMode);
