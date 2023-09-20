import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import IAICollapse from 'common/components/IAICollapse';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useFeatureStatus } from '../../system/hooks/useFeatureStatus';
import ParamDynamicPromptsMaxPrompts from './ParamDynamicPromptsMaxPrompts';
import ParamDynamicPromptsPreview from './ParamDynamicPromptsPreview';
import ParamDynamicPromptsSeedBehaviour from './ParamDynamicPromptsSeedBehaviour';

const ParamDynamicPromptsCollapse = () => {
  const { t } = useTranslation();
  const selectActiveLabel = useMemo(
    () =>
      createSelector(stateSelector, ({ dynamicPrompts }) => {
        const count = dynamicPrompts.prompts.length;
        if (count === 1) {
          return t('dynamicPrompts.promptsWithCount_one', {
            count,
          });
        } else {
          return t('dynamicPrompts.promptsWithCount_other', {
            count,
          });
        }
      }),
    [t]
  );
  const activeLabel = useAppSelector(selectActiveLabel);

  const isDynamicPromptingEnabled =
    useFeatureStatus('dynamicPrompting').isFeatureEnabled;

  if (!isDynamicPromptingEnabled) {
    return null;
  }

  return (
    <IAICollapse
      label={t('dynamicPrompts.dynamicPrompts')}
      activeLabel={activeLabel}
    >
      <Flex sx={{ gap: 2, flexDir: 'column' }}>
        <ParamDynamicPromptsSeedBehaviour />
        <ParamDynamicPromptsPreview />
        <ParamDynamicPromptsMaxPrompts />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamDynamicPromptsCollapse);
