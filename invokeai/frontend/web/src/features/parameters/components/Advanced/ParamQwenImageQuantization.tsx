import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { qwenImageQuantizationChanged, selectQwenImageQuantization } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const QUANTIZATION_OPTIONS: ComboboxOption[] = [
  { value: 'none', label: 'None (bf16)' },
  { value: 'int8', label: '8-bit (int8)' },
  { value: 'nf4', label: '4-bit (nf4)' },
];

const isValidQuantization = (value: string | undefined): value is 'none' | 'int8' | 'nf4' => {
  return value === 'none' || value === 'int8' || value === 'nf4';
};

const ParamQwenImageQuantization = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const quantization = useAppSelector(selectQwenImageQuantization);

  const value = useMemo(() => QUANTIZATION_OPTIONS.find((o) => o.value === quantization), [quantization]);

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
      <Combobox value={value} options={QUANTIZATION_OPTIONS} onChange={onChange} />
    </FormControl>
  );
});

ParamQwenImageQuantization.displayName = 'ParamQwenImageQuantization';

export default ParamQwenImageQuantization;
