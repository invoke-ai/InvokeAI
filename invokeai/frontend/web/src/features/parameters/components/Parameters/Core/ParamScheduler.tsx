import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import { setScheduler } from 'features/parameters/store/generationSlice';
import { SCHEDULER_LABEL_MAP } from 'features/parameters/types/constants';
import { ParameterScheduler } from 'features/parameters/types/parameterSchemas';
import { map } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(
  [stateSelector],
  ({ ui, generation }) => {
    const { scheduler } = generation;
    const { favoriteSchedulers: enabledSchedulers } = ui;

    const data = map(SCHEDULER_LABEL_MAP, (label, name) => ({
      value: name,
      label: label,
      group: enabledSchedulers.includes(name as ParameterScheduler)
        ? 'Favorites'
        : undefined,
    })).sort((a, b) => a.label.localeCompare(b.label));

    return {
      scheduler,
      data,
    };
  }
);

const ParamScheduler = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { scheduler, data } = useAppSelector(selector);

  const handleChange = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }
      dispatch(setScheduler(v as ParameterScheduler));
    },
    [dispatch]
  );

  return (
    <IAIInformationalPopover feature="paramScheduler">
      <IAIMantineSearchableSelect
        label={t('parameters.scheduler')}
        value={scheduler}
        data={data}
        onChange={handleChange}
      />
    </IAIInformationalPopover>
  );
};

export default memo(ParamScheduler);
