import { Badge } from '@chakra-ui/react';
import { useQueueItemProgress } from '@features/queue/react';
import { useTranslation } from 'react-i18next';

/**
 * Leaf subscriber for live generation state, kept out of the frame component
 * so per-step progress events only re-render this sliver.
 */

const formatPercent = (percentage: number): string => `${Math.round(percentage * 100)}%`;

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
