import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  selectRefinerScheduler,
  setRefinerScheduler,
  useParamsDispatch,
} from 'features/controlLayers/store/paramsSlice';
import { SCHEDULER_OPTIONS } from 'features/parameters/types/constants';
import { isParameterScheduler } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSDXLRefinerScheduler = () => {
  const dispatchParams = useParamsDispatch();
  const { t } = useTranslation();
  const refinerScheduler = useAppSelector(selectRefinerScheduler);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterScheduler(v?.value)) {
        return;
      }
      dispatchParams(setRefinerScheduler, v.value);
    },
    [dispatchParams]
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
