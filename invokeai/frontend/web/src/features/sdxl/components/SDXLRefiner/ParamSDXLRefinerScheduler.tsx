import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { SCHEDULER_OPTIONS } from 'features/parameters/types/constants';
import { isParameterScheduler } from 'features/parameters/types/parameterSchemas';
import { setRefinerScheduler } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSDXLRefinerScheduler = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const refinerScheduler = useAppSelector((s) => s.sdxl.refinerScheduler);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterScheduler(v?.value)) {
        return;
      }
      dispatch(setRefinerScheduler(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => SCHEDULER_OPTIONS.find((o) => o.value === refinerScheduler), [refinerScheduler]);

  return (
    <FormControl>
      <FormLabel>{t('sdxl.scheduler')}</FormLabel>
      <Combobox value={value} options={SCHEDULER_OPTIONS} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamSDXLRefinerScheduler);
