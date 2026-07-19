import { Progress, Stack, Text } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

/**
 * Thin denoising progress bar with a "step 12/30 · 39%" caption. `percentage` is
 * 0..1 (null → indeterminate). The caption uses tabular numerals so the live
 * step count and percent don't shift the layout as they tick.
 */
export const QueueStepProgress = ({ message, percentage }: { message: string; percentage: number | null }) => {
  const { t } = useTranslation();
  const percentLabel = percentage !== null ? `${Math.round(percentage * 100)}%` : '';
  const caption = [message.trim(), percentLabel].filter(Boolean).join(' · ');

  return (
    <Stack gap="1.5" w="full">
      <Progress.Root
        aria-label={t('widgets.queue.itemProgress')}
        colorPalette="accent"
        max={1}
        size="xs"
        value={percentage}
        w="full"
      >
        <Progress.Track rounded="full">
          <Progress.Range rounded="full" transition="width var(--wb-motion-duration-fast) ease" />
        </Progress.Track>
      </Progress.Root>
      {caption ? (
        <Text color="fg.subtle" fontSize="2xs" fontVariantNumeric="tabular-nums">
          {caption}
        </Text>
      ) : null}
    </Stack>
  );
};
