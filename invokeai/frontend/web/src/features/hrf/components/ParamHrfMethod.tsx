import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setHrfMethod } from 'features/hrf/store/hrfSlice';
import { isParameterHRFMethod } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const options: ComboboxOption[] = [
  { label: 'ESRGAN', value: 'ESRGAN' },
  { label: 'bilinear', value: 'bilinear' },
];

const ParamHrfMethodSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const hrfMethod = useAppSelector((s) => s.hrf.hrfMethod);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterHRFMethod(v?.value)) {
        return;
      }
      dispatch(setHrfMethod(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => options.find((o) => o.value === hrfMethod), [hrfMethod]);

  return (
    <FormControl>
      <FormLabel>{t('hrf.upscaleMethod')}</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamHrfMethodSelect);
