import { Progress, Stack, Text } from '@chakra-ui/react';
import { llmTaskProgressStore } from '@features/generation/data/llmTaskProgress';
import { useTranslation } from 'react-i18next';

export const LLMTaskProgressDisplay = ({ taskId }: { taskId: string | null }) => {
  const { t } = useTranslation();
  const progress = llmTaskProgressStore.useValue(taskId ?? '');

  if (!progress) {
    return null;
  }

  const phaseLabel =
    progress.phase === 'loading_model'
      ? t('widgets.generate.llmTaskLoadingModel')
      : t('widgets.generate.llmTaskGenerating');
  const tokenLabel =
    progress.current_tokens !== null && progress.total_tokens !== null
      ? `${progress.current_tokens} / ${progress.total_tokens}`
      : null;

  return (
    <Stack gap="1.5" w="full">
      <Progress.Root colorPalette="accent" max={1} size="xs" value={progress.percentage} w="full">
        <Progress.Track aria-label={t('widgets.generate.llmTaskProgress')} rounded="full">
          <Progress.Range rounded="full" transition="width var(--wb-motion-duration-fast) ease" />
        </Progress.Track>
      </Progress.Root>
      <Text color="fg.muted" fontSize="2xs" fontVariantNumeric="tabular-nums">
        {[phaseLabel, tokenLabel].filter(Boolean).join(' · ')}
      </Text>
    </Stack>
  );
};
