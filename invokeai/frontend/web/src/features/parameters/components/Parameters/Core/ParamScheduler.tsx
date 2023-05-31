import { createSelector } from '@reduxjs/toolkit';
import { Scheduler } from 'app/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICustomSelect from 'common/components/IAICustomSelect';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { setScheduler } from 'features/parameters/store/generationSlice';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [uiSelector, generationSelector],
  (ui, generation) => {
    // TODO: DPMSolverSinglestepScheduler is fixed in https://github.com/huggingface/diffusers/pull/3413
    // but we need to wait for the next release before removing this special handling.
    const allSchedulers = ui.schedulers.filter((scheduler) => {
      return !['dpmpp_2s'].includes(scheduler);
    });

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
    (v: string | null | undefined) => {
      if (!v) {
        return;
      }
      dispatch(setScheduler(v as Scheduler));
    },
    [dispatch]
  );

  return (
    <IAICustomSelect
      label={t('parameters.scheduler')}
      selectedItem={scheduler}
      setSelectedItem={handleChange}
      items={allSchedulers}
      withCheckIcon
    />
  );
};

export default memo(ParamScheduler);
