import type { GalleryImage } from '@workbench/gallery/api';
import type { ImageActions } from '@workbench/image-actions';
import type { GeneratedImageContract } from '@workbench/types';

import { HStack, Stack, Text } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { ChevronLeftIcon, ChevronRightIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { PreviewLiveStatusLine } from './PreviewLiveReadout';
import { PreviewMetadataPanel } from './PreviewMetadataPanel';

/**
 * The preview's slim status bar: board position and dimensions (or live
 * progress) on one quiet row with prev/next, plus the Details expander.
 * Identity (board / image name) lives in the widget header; image actions
 * live in the header's actions slot — never here.
 */
export const PreviewFooter = ({
  actionImage,
  actions,
  boardImageCount,
  image,
  isLive,
  isLoadingBoard,
  isMetadataOpen,
  liveQueueItemId,
  onNext,
  onPrevious,
  onToggleMetadata,
  selectedIndex,
}: {
  /** The selected image with board/star context, for the metadata/recall panel. */
  actionImage: GalleryImage | null;
  actions: ImageActions;
  boardImageCount: number;
  image: GeneratedImageContract;
  isLive: boolean;
  isLoadingBoard: boolean;
  isMetadataOpen: boolean;
  /** The live run's local queue item id, when a progress readout should replace the dimensions. */
  liveQueueItemId?: string | null;
  onNext: () => void;
  onPrevious: () => void;
  onToggleMetadata: () => void;
  selectedIndex: number;
}) => {
  const { t } = useTranslation();
  const positionLabel = isLoadingBoard
    ? t('widgets.preview.loadingBoard')
    : selectedIndex === -1
      ? t('widgets.preview.imageCount', { count: boardImageCount })
      : t('common.countOfTotal', { count: selectedIndex + 1, total: boardImageCount });

  return (
    <Stack borderWidth="1px" borderColor="border.subtle" gap="2" p="3" rounded="lg">
      <HStack align="center" justify="space-between">
        <HStack gap="1" minW="0">
          <Text color="fg.subtle" fontSize="2xs" truncate>
            {positionLabel}
          </Text>
          <Text color="fg.subtle" flexShrink={0} fontSize="2xs">
            ·
          </Text>
          {isLive && typeof liveQueueItemId === 'string' ? (
            <PreviewLiveStatusLine queueItemId={liveQueueItemId} />
          ) : (
            <Text color="fg.subtle" flexShrink={0} fontSize="2xs">
              {isLive ? t('common.generating') : `${image.width} × ${image.height}`}
            </Text>
          )}
        </HStack>
        <HStack flexShrink={0} gap="1">
          <Button
            aria-label={t('widgets.preview.previousImageInBoard')}
            disabled={selectedIndex <= 0}
            size="2xs"
            variant="outline"
            onClick={onPrevious}
          >
            <ChevronLeftIcon />
          </Button>
          <Button
            aria-label={t('widgets.preview.nextImageInBoard')}
            disabled={selectedIndex === -1 || selectedIndex >= boardImageCount - 1}
            size="2xs"
            variant="outline"
            onClick={onNext}
          >
            <ChevronRightIcon />
          </Button>
        </HStack>
      </HStack>
      {actionImage ? (
        <PreviewMetadataPanel
          actions={actions}
          image={actionImage}
          isOpen={isMetadataOpen}
          onToggle={onToggleMetadata}
        />
      ) : null}
    </Stack>
  );
};
