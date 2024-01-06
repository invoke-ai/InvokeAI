import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type { InvSelectOnChange } from 'common/components/InvSelect/types';
import { setScheduler } from 'features/parameters/store/generationSlice';
import { SCHEDULER_OPTIONS } from 'features/parameters/types/constants';
import { isParameterScheduler } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamScheduler = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const scheduler = useAppSelector((s) => s.generation.scheduler);

  const onChange = useCallback<InvSelectOnChange>(
    (v) => {
      if (!isParameterScheduler(v?.value)) {
        return;
      }
      dispatch(setScheduler(v.value));
    },
    [dispatch]
  );

  const value = useMemo(
    () => SCHEDULER_OPTIONS.find((o) => o.value === scheduler),
    [scheduler]
  );

  return (
    <InvControl label={t('parameters.scheduler')} feature="paramScheduler">
      <InvSelect
        value={value}
        options={SCHEDULER_OPTIONS}
        onChange={onChange}
      />
    </InvControl>
  );
};

export default memo(ParamScheduler);
