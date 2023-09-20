import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISwitch from 'common/components/IAISwitch';
import { memo, useCallback } from 'react';
import { isEnabledToggled } from '../store/dynamicPromptsSlice';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { isEnabled } = state.dynamicPrompts;

    return { isEnabled };
  },
  defaultSelectorOptions
);

const ParamDynamicPromptsToggle = () => {
  const dispatch = useAppDispatch();
  const { isEnabled } = useAppSelector(selector);
  const { t } = useTranslation();

  const handleToggleIsEnabled = useCallback(() => {
    dispatch(isEnabledToggled());
  }, [dispatch]);

  return (
    <IAISwitch
      label={t('prompt.enableDynamicPrompts')}
      isChecked={isEnabled}
      onChange={handleToggleIsEnabled}
    />
  );
};

export default memo(ParamDynamicPromptsToggle);
