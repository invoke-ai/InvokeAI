import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useFeatureStatus } from '../../system/hooks/useFeatureStatus';
import ParamDynamicPromptsMaxPrompts from './ParamDynamicPromptsMaxPrompts';
import ParamDynamicPromptsPreview from './ParamDynamicPromptsPreview';
import ParamDynamicPromptsSeedBehaviour from './ParamDynamicPromptsSeedBehaviour';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { prompts } = state.dynamicPrompts;
    return {
      activeLabel: `${prompts.length} Prompt${prompts.length !== 1 ? 's' : ''}`,
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
        <ParamDynamicPromptsSeedBehaviour />
        <ParamDynamicPromptsPreview />
        <ParamDynamicPromptsMaxPrompts />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamDynamicPromptsCollapse);
