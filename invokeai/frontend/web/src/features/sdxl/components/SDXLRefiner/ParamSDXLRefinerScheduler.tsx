import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectRefinerScheduler, setRefinerScheduler } from 'features/controlLayers/store/paramsSlice';
import { SCHEDULER_OPTIONS } from 'features/parameters/types/constants';
import { isParameterScheduler } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSDXLRefinerScheduler = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const refinerScheduler = useAppSelector(selectRefinerScheduler);

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
      <InformationalPopover feature="refinerScheduler">
        <FormLabel>{t('sdxl.scheduler')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={SCHEDULER_OPTIONS} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamSDXLRefinerScheduler);
