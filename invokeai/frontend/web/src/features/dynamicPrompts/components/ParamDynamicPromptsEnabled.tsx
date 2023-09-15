import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISwitch from 'common/components/IAISwitch';
import { memo, useCallback } from 'react';
import { isEnabledToggled } from '../store/dynamicPromptsSlice';
import { DynamicPromptsTogglePopover } from 'features/informationalPopovers/components/dynamicPromptsToggle';

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

  const handleToggleIsEnabled = useCallback(() => {
    dispatch(isEnabledToggled());
  }, [dispatch]);

  return (
    <DynamicPromptsTogglePopover>
      <IAISwitch
        label="Enable Dynamic Prompts"
        isChecked={isEnabled}
        onChange={handleToggleIsEnabled}
      />
    </DynamicPromptsTogglePopover>
  );
};

export default memo(ParamDynamicPromptsToggle);
