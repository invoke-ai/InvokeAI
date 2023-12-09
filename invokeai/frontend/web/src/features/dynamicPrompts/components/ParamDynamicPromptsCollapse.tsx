import { Flex } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import IAICollapse from 'common/components/IAICollapse';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import ParamDynamicPromptsMaxPrompts from './ParamDynamicPromptsMaxPrompts';
import ParamDynamicPromptsPreview from './ParamDynamicPromptsPreview';
import ParamDynamicPromptsSeedBehaviour from './ParamDynamicPromptsSeedBehaviour';

const ParamDynamicPromptsCollapse = () => {
  const { t } = useTranslation();
  const selectActiveLabel = useMemo(
    () =>
      createMemoizedSelector(stateSelector, ({ dynamicPrompts }) => {
        const count = dynamicPrompts.prompts.length;
        if (count > 1) {
          return t('dynamicPrompts.promptsWithCount_other', {
            count,
          });
        }

        return;
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
        <ParamDynamicPromptsPreview />
        <ParamDynamicPromptsSeedBehaviour />
        <ParamDynamicPromptsMaxPrompts />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamDynamicPromptsCollapse);
