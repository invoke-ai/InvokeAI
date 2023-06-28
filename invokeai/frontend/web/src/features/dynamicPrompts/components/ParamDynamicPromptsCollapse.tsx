import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import { useCallback } from 'react';
import { isEnabledToggled } from '../store/slice';
import ParamDynamicPromptsMaxPrompts from './ParamDynamicPromptsMaxPrompts';
import ParamDynamicPromptsCombinatorial from './ParamDynamicPromptsCombinatorial';
import { Flex } from '@chakra-ui/react';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { isEnabled } = state.dynamicPrompts;

    return { isEnabled };
  },
  defaultSelectorOptions
);

const ParamDynamicPromptsCollapse = () => {
  const dispatch = useAppDispatch();
  const { isEnabled } = useAppSelector(selector);

  const handleToggleIsEnabled = useCallback(() => {
    dispatch(isEnabledToggled());
  }, [dispatch]);

  return (
    <IAICollapse
      isOpen={isEnabled}
      onToggle={handleToggleIsEnabled}
      label="Dynamic Prompts"
      withSwitch
    >
      <Flex sx={{ gap: 2, flexDir: 'column' }}>
        <ParamDynamicPromptsCombinatorial />
        <ParamDynamicPromptsMaxPrompts />
      </Flex>
    </IAICollapse>
  );
};

export default ParamDynamicPromptsCollapse;
