import type { GalleryImage } from '@workbench/gallery/api';
import type { ImageActions } from '@workbench/image-actions';
import type { GeneratedImageContract } from '@workbench/types';

import { Badge, Box, HStack, Stack, Text } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { ChevronLeftIcon, ChevronRightIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import type { PreviewDensity } from './previewDensity';

import { PreviewActionStrip } from './PreviewActionStrip';

/**
 * The preview's info card: board context ("N of M"), the action strip,
 * prev/next navigation and the image identity rows. Later phases add the step
 * message and metadata rail here — never in the widget shell.
 */
export const PreviewFooter = ({
  actionImage,
  actions,
  boardImageCount,
  boardName,
  density,
  image,
  isLive,
  isLoadingBoard,
  onNext,
  onPrevious,
  selectedIndex,
}: {
  actionImage: GalleryImage | null;
  actions: ImageActions;
  boardImageCount: number;
  boardName: string;
  density: PreviewDensity;
  image: GeneratedImageContract;
  isLive: boolean;
  isLoadingBoard: boolean;
  onNext: () => void;
  onPrevious: () => void;
  selectedIndex: number;
}) => {
  const { t } = useTranslation();

  return (
    <Stack borderWidth="1px" borderColor="border.subtle" gap="2" p="3" rounded="lg">
      <HStack align="center" justify="space-between">
        <HStack gap="2" minW="0">
          <Badge flexShrink={0} size="xs" variant="subtle">
            {boardName}
          </Badge>
          <Text color="fg.subtle" fontSize="2xs" truncate>
            {isLoadingBoard
              ? t('widgets.preview.loadingBoard')
              : selectedIndex === -1
                ? t('widgets.preview.imageCount', { count: boardImageCount })
                : t('common.countOfTotal', { count: selectedIndex + 1, total: boardImageCount })}
          </Text>
        </HStack>
        <HStack flexShrink={0} gap="1">
          {actionImage ? (
            <>
              <PreviewActionStrip actions={actions} density={density} image={actionImage} />
              <Box bg="border.subtle" flexShrink={0} h="4" mx="0.5" w="1px" />
            </>
          ) : null}
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
      <Stack>
        <Text fontSize="xs" fontWeight="800" truncate>
          {image.imageName}
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          {t('widgets.preview.sourceRun', { id: image.sourceQueueItemId })}
        </Text>
        <Text color="fg.subtle" flexShrink={0} fontSize="2xs">
          {isLive ? t('common.generating') : `${image.width} x ${image.height}`}
        </Text>
      </Stack>
    </Stack>
  );
};
