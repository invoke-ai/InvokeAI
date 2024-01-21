import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { setMaskBlurMethod } from 'features/parameters/store/generationSlice';
import { isParameterMaskBlurMethod } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const options: ComboboxOption[] = [
  { label: 'Box Blur', value: 'box' },
  { label: 'Gaussian Blur', value: 'gaussian' },
];

const ParamMaskBlurMethod = () => {
  const maskBlurMethod = useAppSelector((s) => s.generation.maskBlurMethod);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterMaskBlurMethod(v?.value)) {
        return;
      }
      dispatch(setMaskBlurMethod(v.value));
    },
    [dispatch]
  );

  const value = useMemo(
    () => options.find((o) => o.value === maskBlurMethod),
    [maskBlurMethod, options]
  );

  return (
    <FormControl>
      <InformationalPopover feature="compositingBlurMethod">
        <FormLabel>{t('parameters.maskBlurMethod')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} onChange={onChange} options={options} />
    </FormControl>
  );
};

export default memo(ParamMaskBlurMethod);
