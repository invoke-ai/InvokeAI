import { Badge, Box, Text } from '@chakra-ui/react';
import { useQueueItemProgress, type QueueItemProgress } from '@features/queue/react';
import { useMemo, type CSSProperties } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Leaf subscribers for live generation state, kept out of the frame/footer
 * components so per-step progress events only re-render these slivers.
 */

const formatPercent = (percentage: number): string => `${Math.round(percentage * 100)}%`;

const getStepMessage = (progress: QueueItemProgress | null, generatingLabel: string): string => {
  if (!progress) {
    return generatingLabel;
  }

  const message = progress.message || generatingLabel;

  if (progress.totalItemCount && progress.totalItemCount > 1 && progress.activeItemIndex) {
    return `${progress.activeItemIndex}/${progress.totalItemCount} · ${message}`;
  }

  return message;
};

/** Badge + 2px hairline progress bar drawn over the fitted frame while live. */
export const PreviewLiveOverlay = ({ queueItemId }: { queueItemId: string }) => {
  const { t } = useTranslation();
  const progress = useQueueItemProgress(queueItemId);
  const percentage = progress?.percentage ?? null;
  const fillStyle = useMemo<CSSProperties>(
    () => ({ width: percentage === null ? '100%' : `${percentage * 100}%` }),
    [percentage]
  );

  return (
    <>
      <Badge left="2" pointerEvents="none" position="absolute" size="xs" top="2" variant="solid">
        {percentage === null ? t('common.generating') : `${t('common.generating')} · ${formatPercent(percentage)}`}
      </Badge>
      <Box bottom="0" h="2px" left="0" pointerEvents="none" position="absolute" right="0">
        <Box
          bg="accent.solid"
          h="full"
          opacity={percentage === null ? 0.35 : 1}
          style={fillStyle}
          transitionDuration="var(--wb-motion-duration-fast)"
          transitionProperty="width"
          transitionTimingFunction="ease"
        />
      </Box>
    </>
  );
};

/** The footer's live status line ("2/4 · Denoising"), replacing the dimensions row while generating. */
export const PreviewLiveStatusLine = ({ queueItemId }: { queueItemId: string }) => {
  const { t } = useTranslation();
  const progress = useQueueItemProgress(queueItemId);

  return (
    <Text color="fg.subtle" flexShrink={0} fontSize="2xs">
      {getStepMessage(progress, t('common.generating'))}
    </Text>
  );
};
