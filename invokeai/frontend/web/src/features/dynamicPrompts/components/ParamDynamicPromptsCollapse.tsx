import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useFeatureStatus } from '../../system/hooks/useFeatureStatus';
import ParamDynamicPromptsCombinatorial from './ParamDynamicPromptsCombinatorial';
import ParamDynamicPromptsToggle from './ParamDynamicPromptsEnabled';
import ParamDynamicPromptsMaxPrompts from './ParamDynamicPromptsMaxPrompts';
import ParamDynamicPromptsPreview from './ParamDynamicPromptsPreview';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { isEnabled, prompts } = state.dynamicPrompts;

    return {
      activeLabel: isEnabled
        ? `${prompts.length} Prompt${prompts.length > 1 ? 's' : ''}`
        : undefined,
    };
  },
  defaultSelectorOptions
);

const ParamDynamicPromptsCollapse = () => {
  const { activeLabel } = useAppSelector(selector);
  const { t } = useTranslation();

  const isDynamicPromptingEnabled =
    useFeatureStatus('dynamicPrompting').isFeatureEnabled;

  if (!isDynamicPromptingEnabled) {
    return null;
  }

  return (
    <IAICollapse label={t('prompt.dynamicPrompts')} activeLabel={activeLabel}>
      <Flex sx={{ gap: 2, flexDir: 'column' }}>
        <ParamDynamicPromptsToggle />
        <ParamDynamicPromptsPreview />
        <ParamDynamicPromptsMaxPrompts />
        <ParamDynamicPromptsCombinatorial />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamDynamicPromptsCollapse);
