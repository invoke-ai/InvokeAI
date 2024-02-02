import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
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

  const value = useMemo(() => options.find((o) => o.value === vaePrecision), [vaePrecision]);

  return (
    <FormControl w="14rem" flexShrink={0}>
      <InformationalPopover feature="paramVAEPrecision">
        <FormLabel>{t('modelManager.vaePrecision')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={options} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamVAEModelSelect);
