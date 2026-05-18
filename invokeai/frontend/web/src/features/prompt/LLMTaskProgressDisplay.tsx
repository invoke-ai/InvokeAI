import { Flex, Progress, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { round } from 'es-toolkit/compat';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { $llmTaskStates } from 'services/events/stores';

type Props = {
  taskId: string | null;
};

export const LLMTaskProgressDisplay = memo(({ taskId }: Props) => {
  const { t } = useTranslation();
  const allStates = useStore($llmTaskStates);
  const state = useMemo(() => (taskId ? allStates[taskId] : undefined), [allStates, taskId]);

  if (!taskId || !state || state.status === 'complete') {
    return null;
  }

  if (state.status === 'error') {
    return (
      <Text fontSize="xs" color="error.300">
        {state.error}
      </Text>
    );
  }

  const { phase, percentage, current_tokens, total_tokens } = state.payload;
  const label = phase === 'loading_model' ? t('prompt.llmTaskLoadingModel') : t('prompt.llmTaskGenerating');
  const isIndeterminate = phase === 'loading_model' || percentage === null;
  const pct = percentage !== null ? round(percentage * 100, 1) : 0;

  return (
    <Flex flexDir="column" gap={1}>
      <Flex justifyContent="space-between" alignItems="center">
        <Text fontSize="xs" color="base.300">
          {label}
        </Text>
        {phase === 'generating' && current_tokens !== null && total_tokens !== null ? (
          <Text fontSize="xs" color="base.400">
            {current_tokens} / {total_tokens}
          </Text>
        ) : null}
      </Flex>
      <Progress size="xs" value={pct} isIndeterminate={isIndeterminate} colorScheme="invokeBlue" borderRadius="base" />
    </Flex>
  );
});

LLMTaskProgressDisplay.displayName = 'LLMTaskProgressDisplay';
