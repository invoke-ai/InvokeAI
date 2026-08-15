import type { GalleryImage, GalleryItem } from '@features/gallery';
import type { ImageActions } from '@workbench/image-actions';

import { HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { formatGalleryVideoDuration } from '@features/gallery/contracts';
import { Button } from '@platform/ui';
import { ChevronLeftIcon, ChevronRightIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { PreviewMetadataPanel } from './PreviewMetadataPanel';

/**
 * What the footer is describing: a selected gallery item, or the image slot a
 * live generation is filling. The live shape carries the slot's requested
 * output dimensions — the denoise preview frames arrive at other sizes, and a
 * dimensions readout that ticks while rendering reads as noise.
 */
export type PreviewFooterMedia =
  | { actionImage: GalleryImage | null; actions: ImageActions; item: GalleryItem; kind: 'item' }
  | { height: number; kind: 'live'; width: number };

/**
 * The preview's slim status bar: board position and dimensions on one quiet
 * row with prev/next, plus the Details expander. Identity (board / image name)
 * lives in the widget header; image actions live in the header's actions slot
 * — never here.
 *
 * It floats over the dot-grid surface rather than partitioning it, so it
 * carries its own opaque panel fill — the grid runs behind and past it.
 *
 * It renders while a generation is live too, with the same geometry: the
 * moment denoising finishes is exactly when the fitted frame must not move,
 * so the bar never appears or disappears across that boundary.
 */
export const PreviewFooter = ({
  boardItemCount,
  isLoadingBoard,
  isMetadataOpen,
  media,
  onNext,
  onPrevious,
  onToggleMetadata,
  selectedIndex,
}: {
  boardItemCount: number;
  isLoadingBoard: boolean;
  isMetadataOpen: boolean;
  media: PreviewFooterMedia;
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
  const item = media.kind === 'item' ? media.item : null;
  const fps =
    item?.kind === 'video' && item.fps !== undefined
      ? new Intl.NumberFormat(i18n.language, { maximumFractionDigits: 3 }).format(item.fps)
      : null;
  const { height, width } = media.kind === 'item' ? media.item : media;

  return (
    <Stack bg="bg.subtle" borderColor="border.subtle" borderWidth="1px" gap="2" minW="0" p="3" rounded="md" shadow="sm">
      <HStack align="center" justify="space-between">
        <HStack gap="1" minW="0">
          <Text color="fg.muted" fontSize="2xs" fontVariantNumeric="tabular-nums" truncate>
            {positionLabel}
          </Text>
          <Text color="fg.subtle" flexShrink={0} fontSize="2xs">
            ·
          </Text>
          <Text color="fg.muted" flexShrink={0} fontSize="2xs" fontVariantNumeric="tabular-nums">
            {width} × {height}
            {item?.kind === 'video'
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
      {media.kind === 'item' ? (
        <PreviewMetadataPanel
          actions={media.actions}
          image={media.actionImage}
          isOpen={isMetadataOpen}
          item={media.item}
          onToggle={onToggleMetadata}
        />
      ) : (
        // The Details row an in-flight image cannot answer yet, at the exact
        // geometry of the live one, so completion swaps content — never layout.
        <HStack aria-disabled="true" color="fg.subtle" gap="1" w="fit-content">
          <Icon as={ChevronRightIcon} boxSize="3" />
          <Text fontSize="2xs" fontWeight="700" textTransform="uppercase">
            {t('widgets.preview.details')}
          </Text>
        </HStack>
      )}
    </Stack>
  );
};
