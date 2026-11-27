import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { qwenImageQuantizationChanged, selectQwenImageQuantization } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const isValidQuantization = (value: string | undefined): value is 'none' | 'int8' | 'nf4' => {
  return value === 'none' || value === 'int8' || value === 'nf4';
};

const ParamQwenImageQuantization = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const quantization = useAppSelector(selectQwenImageQuantization);

  const options = useMemo<ComboboxOption[]>(
    () => [
      { value: 'none', label: t('modelManager.qwenImageQuantizationNone') },
      { value: 'int8', label: t('modelManager.qwenImageQuantizationInt8') },
      { value: 'nf4', label: t('modelManager.qwenImageQuantizationNf4') },
    ],
    [t]
  );

  const value = useMemo(() => options.find((o) => o.value === quantization), [options, quantization]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isValidQuantization(v?.value)) {
        return;
      }
      dispatch(qwenImageQuantizationChanged(v.value));
    },
    [dispatch]
  );

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.qwenImageQuantization')}</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} />
    </FormControl>
  );
});

ParamQwenImageQuantization.displayName = 'ParamQwenImageQuantization';

export default ParamQwenImageQuantization;
