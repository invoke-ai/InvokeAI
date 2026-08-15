import { Badge, Text } from '@chakra-ui/react';
import { useQueueItemProgress, type QueueItemProgress } from '@features/queue/react';
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

/**
 * The live badge drawn over the fitted frame.
 *
 * No bar under it any more: the denoising image *is* the progress here — you can
 * watch it resolve — and the top bar's rail already carries the quantified
 * version at viewport width. A second hairline across the image only competed
 * with the thing it was describing.
 */
export const PreviewLiveOverlay = ({ queueItemId }: { queueItemId: string }) => {
  const { t } = useTranslation();
  const progress = useQueueItemProgress(queueItemId);
  const percentage = progress?.percentage ?? null;

  return (
    <Badge left="2" pointerEvents="none" position="absolute" size="xs" top="2" variant="solid">
      {percentage === null ? t('common.generating') : `${t('common.generating')} · ${formatPercent(percentage)}`}
    </Badge>
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
