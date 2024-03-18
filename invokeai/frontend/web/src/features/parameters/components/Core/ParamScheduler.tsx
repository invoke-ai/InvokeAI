import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { setScheduler } from 'features/parameters/store/generationSlice';
import { SCHEDULER_OPTIONS } from 'features/parameters/types/constants';
import { isParameterScheduler } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamScheduler = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const scheduler = useAppSelector((s) => s.generation.scheduler);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterScheduler(v?.value)) {
        return;
      }
      dispatch(setScheduler(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => SCHEDULER_OPTIONS.find((o) => o.value === scheduler), [scheduler]);

  return (
    <FormControl>
      <InformationalPopover feature="paramScheduler">
        <FormLabel>{t('parameters.scheduler')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={SCHEDULER_OPTIONS} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamScheduler);
