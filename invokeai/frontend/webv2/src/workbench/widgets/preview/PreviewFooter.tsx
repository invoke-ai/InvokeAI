import type { GalleryImage, GalleryItem } from '@features/gallery';
import type { ImageActions } from '@workbench/image-actions';

import { HStack, Stack, Text } from '@chakra-ui/react';
import { formatGalleryVideoDuration } from '@features/gallery/contracts';
import { Button } from '@platform/ui';
import { ChevronLeftIcon, ChevronRightIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

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
  boardItemCount,
  isLoadingBoard,
  isMetadataOpen,
  item,
  onNext,
  onPrevious,
  onToggleMetadata,
  selectedIndex,
}: {
  /** The selected image with board/star context, for the metadata/recall panel. */
  actionImage: GalleryImage | null;
  actions: ImageActions;
  boardItemCount: number;
  isLoadingBoard: boolean;
  isMetadataOpen: boolean;
  item: GalleryItem;
  onNext: () => void;
  onPrevious: () => void;
  onToggleMetadata: () => void;
  selectedIndex: number;
}) => {
  const { i18n, t } = useTranslation();
  const positionLabel = isLoadingBoard
    ? t('widgets.preview.loadingBoard')
    : selectedIndex === -1
      ? t('widgets.preview.itemCount', { count: boardItemCount })
      : t('common.countOfTotal', { count: selectedIndex + 1, total: boardItemCount });
  const fps =
    item.kind === 'video' && item.fps !== undefined
      ? new Intl.NumberFormat(i18n.language, { maximumFractionDigits: 3 }).format(item.fps)
      : null;

  return (
    <Stack borderWidth="1px" borderColor="border.subtle" gap="2" p="3" rounded="lg">
      <HStack align="center" justify="space-between">
        <HStack gap="1" minW="0">
          <Text color="fg.subtle" fontSize="2xs" fontVariantNumeric="tabular-nums" truncate>
            {positionLabel}
          </Text>
          <Text color="fg.subtle" flexShrink={0} fontSize="2xs">
            ·
          </Text>
          <Text color="fg.subtle" flexShrink={0} fontSize="2xs" fontVariantNumeric="tabular-nums">
            {item.width} × {item.height}
            {item.kind === 'video'
              ? ` · ${t('widgets.preview.videoDuration', {
                  duration: formatGalleryVideoDuration(item.durationSeconds),
                })}${fps === null ? '' : ` · ${t('widgets.preview.framesPerSecond', { count: fps })}`}`
              : ''}
          </Text>
        </HStack>
        <HStack flexShrink={0} gap="1">
          <Button
            aria-label={t('widgets.preview.previousItemInBoard')}
            disabled={selectedIndex <= 0}
            size="2xs"
            variant="outline"
            onClick={onPrevious}
          >
            <ChevronLeftIcon />
          </Button>
          <Button
            aria-label={t('widgets.preview.nextItemInBoard')}
            disabled={selectedIndex === -1 || selectedIndex >= boardItemCount - 1}
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
