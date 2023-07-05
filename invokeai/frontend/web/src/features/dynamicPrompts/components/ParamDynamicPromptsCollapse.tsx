import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import ParamDynamicPromptsCombinatorial from './ParamDynamicPromptsCombinatorial';
import ParamDynamicPromptsToggle from './ParamDynamicPromptsEnabled';
import ParamDynamicPromptsMaxPrompts from './ParamDynamicPromptsMaxPrompts';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { isEnabled } = state.dynamicPrompts;

    return { activeLabel: isEnabled ? 'Enabled' : undefined };
  },
  defaultSelectorOptions
);

const ParamDynamicPromptsCollapse = () => {
  const { activeLabel } = useAppSelector(selector);

  return (
    <IAICollapse label="Dynamic Prompts" activeLabel={activeLabel}>
      <Flex sx={{ gap: 2, flexDir: 'column' }}>
        <ParamDynamicPromptsToggle />
        <ParamDynamicPromptsCombinatorial />
        <ParamDynamicPromptsMaxPrompts />
      </Flex>
    </IAICollapse>
  );
};

export default ParamDynamicPromptsCollapse;
