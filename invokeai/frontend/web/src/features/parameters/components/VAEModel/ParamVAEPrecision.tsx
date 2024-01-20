import type { ComboboxOnChange } from '@invoke-ai/ui';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { vaePrecisionChanged } from 'features/parameters/store/generationSlice';
import { isParameterPrecision } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const options = [
  { label: 'FP16', value: 'fp16' },
  { label: 'FP32', value: 'fp32' },
];

const ParamVAEModelSelect = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const vaePrecision = useAppSelector((s) => s.generation.vaePrecision);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterPrecision(v?.value)) {
        return;
      }

      dispatch(vaePrecisionChanged(v.value));
    },
    [dispatch]
  );

  const value = useMemo(
    () => options.find((o) => o.value === vaePrecision),
    [vaePrecision]
  );

  return (
    <FormControl feature="paramVAEPrecision" w="14rem" flexShrink={0}>
      <FormLabel>{t('modelManager.vaePrecision')}</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamVAEModelSelect);
