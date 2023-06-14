import { createSelector } from '@reduxjs/toolkit';
import { Scheduler } from 'app/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSelect, {
  IAISelectDataType,
} from 'common/components/IAIMantineSelect';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { setScheduler } from 'features/parameters/store/generationSlice';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [uiSelector, generationSelector],
  (ui, generation) => {
    const allSchedulers: string[] = ui.schedulers
      .slice()
      .sort((a, b) => a.localeCompare(b));

    return {
      scheduler: generation.scheduler,
      allSchedulers,
    };
  },
  defaultSelectorOptions
);

const ParamScheduler = () => {
  const { allSchedulers, scheduler } = useAppSelector(selector);

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }
      dispatch(setScheduler(v as Scheduler));
    },
    [dispatch]
  );

  return (
    <IAIMantineSelect
      label={t('parameters.scheduler')}
      value={scheduler}
      data={allSchedulers}
      onChange={handleChange}
    />
  );
};

export default memo(ParamScheduler);
