import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISwitch from 'common/components/IAISwitch';
import { isIPAdapterEnabledChanged } from 'features/controlNet/store/controlNetSlice';
import { ChangeEvent, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { isIPAdapterEnabled } = state.controlNet;

    return { isIPAdapterEnabled };
  },
  defaultSelectorOptions
);

const ParamIPAdapterFeatureToggle = () => {
  const { isIPAdapterEnabled } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(isIPAdapterEnabledChanged(e.target.checked));
    },
    [dispatch]
  );

  return (
    <IAISwitch
      label={t('controlnet.enableIPAdapter')}
      isChecked={isIPAdapterEnabled}
      onChange={handleChange}
      formControlProps={{
        width: '100%',
      }}
    />
  );
};

export default memo(ParamIPAdapterFeatureToggle);
