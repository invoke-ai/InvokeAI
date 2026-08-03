import type { GalleryThumbnailFit } from '@features/gallery/core/settings';

import { Box, Flex, ProgressCircle, Skeleton } from '@chakra-ui/react';
import { useQueueItemProgress, useQueueItemProgressImage } from '@features/queue/react';
import { StreamingImageFrame } from '@platform/ui/streaming-image/StreamingImageFrame';
import { progressImageToStreamingSource } from '@platform/ui/streaming-image/streamingImageSource';
import { useTranslation } from 'react-i18next';

import type { GalleryQueuePlaceholder } from './galleryStateView';

/** Grid tile standing in for an in-flight queue item bound for the current board. */
export const GalleryQueuePlaceholderCell = ({
  antialiasProgressImages,
  fit,
  isSelected,
  placeholder,
  onClick,
}: {
  antialiasProgressImages: boolean;
  fit: GalleryThumbnailFit;
  isSelected: boolean;
  placeholder: GalleryQueuePlaceholder;
  onClick: () => void;
}) => {
  const { t } = useTranslation();
  const progressImage = useQueueItemProgressImage(placeholder.queueItemId, placeholder.itemIndex);
  const progress = useQueueItemProgress(placeholder.queueItemId);
  const isActive = progress?.activeItemIndex === placeholder.itemIndex;
  const percentage = typeof progress?.percentage === 'number' ? Math.round(progress.percentage * 100) : null;

  return (
    <Box aspectRatio={1} minW="0" role="listitem" w="full">
      <Box
        as="button"
        aria-label={t('widgets.preview.showInProgressDiffusion')}
        aria-pressed={isSelected}
        bg="bg"
        borderColor={isSelected ? 'accent.solid' : 'border.subtle'}
        borderWidth={isSelected ? '2px' : '1px'}
        cursor="pointer"
        h="full"
        overflow="hidden"
        position="relative"
        rounded="md"
        w="full"
        onClick={onClick}
      >
        <StreamingImageFrame
          fit={fit === 'aspect' ? 'contain' : 'cover'}
          h="full"
          liveImage={progressImageToStreamingSource(progressImage)}
          shouldAntialiasLiveImage={antialiasProgressImages}
          w="full"
        >
          <Skeleton h="full" w="full" />
        </StreamingImageFrame>
        {isActive ? <GalleryPlaceholderCircularProgress percentage={percentage} /> : null}
      </Box>
    </Box>
  );
};

const GalleryPlaceholderCircularProgress = ({ percentage }: { percentage: number | null }) => {
  const { t } = useTranslation();

  return (
    <Flex
      align="center"
      alignItems="center"
      inset="0"
      justify="center"
      pointerEvents="none"
      position="absolute"
      zIndex="1"
    >
      <ProgressCircle.Root
        aria-label={
          percentage === null
            ? t('widgets.gallery.generationProgress')
            : t('widgets.gallery.generationProgressPercent', { percentage })
        }
        bg="bg/85"
        borderWidth={1}
        p={0.5}
        rounded="full"
        size="xs"
        value={percentage}
      >
        <ProgressCircle.Circle>
          <ProgressCircle.Track />
          <ProgressCircle.Range />
        </ProgressCircle.Circle>
      </ProgressCircle.Root>
    </Flex>
  );
};
