import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setInfillMethod } from 'features/parameters/store/generationSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetAppConfigQuery } from 'services/api/endpoints/appInfo';

const ParamInfillMethod = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const infillMethod = useAppSelector((s) => s.generation.infillMethod);
  const { data: appConfigData } = useGetAppConfigQuery();
  const options = useMemo<ComboboxOption[]>(
    () =>
      appConfigData
        ? appConfigData.infill_methods.map((method) => ({
            label: method,
            value: method,
          }))
        : [],
    [appConfigData]
  );

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v || !options.find((o) => o.value === v.value)) {
        return;
      }
      dispatch(setInfillMethod(v.value));
    },
    [dispatch, options]
  );

  const value = useMemo(
    () => options.find((o) => o.value === infillMethod),
    [options, infillMethod]
  );

  return (
    <FormControl isDisabled={options.length === 0} feature="infillMethod">
      <FormLabel>{t('parameters.infillMethod')}</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamInfillMethod);
