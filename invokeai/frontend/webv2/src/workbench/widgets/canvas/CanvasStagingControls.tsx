/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { CanvasStagingCandidateContract, GeneratedImageContract } from '@workbench/types';

import { Box, HStack, ScrollArea, Stack, Text } from '@chakra-ui/react';
import { Button, IconButton } from '@workbench/components/ui';
import { ChevronDownIcon, ChevronLeftIcon, ChevronRightIcon, ChevronUpIcon, EyeIcon, EyeOffIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

const THUMBNAIL_STRIP_HEIGHT = '5.5rem';

interface CanvasStagingControlsProps {
  areThumbnailsVisible: boolean;
  hasMultipleCandidates: boolean;
  isVisible: boolean;
  pendingImages: CanvasStagingCandidateContract[];
  selectedCandidate: CanvasStagingCandidateContract;
  selectedImageIndex: number;
  onAccept: () => void;
  onCycle: (direction: -1 | 1) => void;
  onDiscardAll: () => void;
  onDiscardSelected: () => void;
  onSelectImage: (imageIndex: number) => void;
  onToggleThumbnails: () => void;
  onToggleVisibility: () => void;
}

export const CanvasStagingControls = ({
  areThumbnailsVisible,
  hasMultipleCandidates,
  isVisible,
  pendingImages,
  selectedCandidate,
  selectedImageIndex,
  onAccept,
  onCycle,
  onDiscardAll,
  onDiscardSelected,
  onSelectImage,
  onToggleThumbnails,
  onToggleVisibility,
}: CanvasStagingControlsProps) => {
  const { t } = useTranslation();

  return (
    <Stack align="center" bottom="2" gap="2" left="2" position="absolute" right="2" zIndex="3">
      <ScrollArea.Root
        h={areThumbnailsVisible ? THUMBNAIL_STRIP_HEIGHT : '0'}
        opacity={areThumbnailsVisible ? 1 : 0}
        pointerEvents={areThumbnailsVisible ? 'auto' : 'none'}
        size="xs"
        transition="height var(--wb-motion-duration-slow) ease, opacity var(--wb-motion-duration-slow) ease"
        variant="hover"
        w="full"
      >
        <ScrollArea.Viewport h="full" w="full">
          <ScrollArea.Content asChild>
            <HStack h="full">
              {pendingImages.map((candidate, index) => (
                <StagingThumbnail
                  key={`${candidate.sourceQueueItemId}-${candidate.imageName}`}
                  candidate={candidate}
                  index={index}
                  isSelected={index === selectedImageIndex}
                  onSelect={() => onSelectImage(index)}
                />
              ))}
            </HStack>
          </ScrollArea.Content>
        </ScrollArea.Viewport>
        <ScrollArea.Scrollbar orientation="horizontal">
          <ScrollArea.Thumb />
        </ScrollArea.Scrollbar>
      </ScrollArea.Root>

      <Stack
        bg="bg.muted"
        borderWidth="1px"
        borderColor="border.emphasized"
        gap="2"
        minW="18rem"
        p="2"
        rounded="lg"
        shadow="lg"
      >
        <HStack justify="space-between">
          <Text fontSize="xs" fontWeight="700">
            {t('widgets.canvas.candidateCount', { current: selectedImageIndex + 1, total: pendingImages.length })}
          </Text>
          <Text color="fg.subtle" fontSize="2xs">
            {selectedCandidate.width} x {selectedCandidate.height}
          </Text>
        </HStack>
        <HStack justify="space-between">
          <HStack gap="1">
            <IconButton
              aria-label={
                areThumbnailsVisible
                  ? t('widgets.canvas.hideStagingThumbnails')
                  : t('widgets.canvas.showStagingThumbnails')
              }
              size="xs"
              variant="outline"
              onClick={onToggleThumbnails}
            >
              {areThumbnailsVisible ? <ChevronDownIcon /> : <ChevronUpIcon />}
            </IconButton>
            <IconButton
              aria-label={t('widgets.canvas.previousStagedCandidate')}
              disabled={!hasMultipleCandidates}
              size="xs"
              variant="outline"
              onClick={() => onCycle(-1)}
            >
              <ChevronLeftIcon />
            </IconButton>
            <IconButton
              aria-label={t('widgets.canvas.nextStagedCandidate')}
              disabled={!hasMultipleCandidates}
              size="xs"
              variant="outline"
              onClick={() => onCycle(1)}
            >
              <ChevronRightIcon />
            </IconButton>
          </HStack>
          <HStack gap="2">
            <IconButton
              aria-label={
                isVisible ? t('widgets.canvas.hideStagedResultPreview') : t('widgets.canvas.showStagedResultPreview')
              }
              size="xs"
              variant="outline"
              onClick={onToggleVisibility}
            >
              {isVisible ? <EyeIcon /> : <EyeOffIcon />}
            </IconButton>
            <Button size="xs" variant="outline" onClick={onDiscardSelected}>
              {t('common.discard')}
            </Button>
            <Button size="xs" variant="outline" onClick={onDiscardAll}>
              {t('common.discardAll')}
            </Button>
            <Button disabled={!isVisible} size="xs" onClick={onAccept}>
              {t('widgets.canvas.acceptToLayer')}
            </Button>
          </HStack>
        </HStack>
      </Stack>
    </Stack>
  );
};

export const EmptyStagingControls = () => {
  const { t } = useTranslation();

  return (
    <Stack
      bg="bg.muted"
      borderWidth="1px"
      borderColor="border.emphasized"
      bottom="4"
      gap="2"
      left="50%"
      minW="18rem"
      p="3"
      position="absolute"
      rounded="lg"
      shadow="lg"
      transform="translateX(-50%)"
      zIndex="3"
    >
      <Text color="fg.subtle" fontSize="2xs" textAlign="center">
        {t('widgets.canvas.emptyStaging')}
      </Text>
    </Stack>
  );
};

const StagingThumbnail = ({
  candidate,
  index,
  isSelected,
  onSelect,
}: {
  candidate: GeneratedImageContract;
  index: number;
  isSelected: boolean;
  onSelect: () => void;
}) => {
  const { t } = useTranslation();

  return (
    <Box
      aria-label={t('widgets.canvas.selectStagedCandidate', { number: index + 1 })}
      as="button"
      bg="bg.emphasized"
      borderWidth="2px"
      borderColor={isSelected ? 'accent.solid' : 'border.subtle'}
      flex="0 0 auto"
      h="full"
      overflow="hidden"
      position="relative"
      rounded="md"
      shadow={isSelected ? '0 0 0 1px {colors.accent.solid}' : 'md'}
      style={{ aspectRatio: '1 / 1' }}
      onClick={onSelect}
    >
      <img
        alt={candidate.imageName}
        src={candidate.thumbnailUrl || candidate.imageUrl}
        style={{ display: 'block', height: '100%', objectFit: 'cover', width: '100%' }}
      />
      <Text
        bg="blackAlpha.700"
        bottom="1"
        color="white"
        fontSize="2xs"
        fontWeight="700"
        left="1"
        px="1.5"
        position="absolute"
        rounded="sm"
      >
        {index + 1}
      </Text>
    </Box>
  );
};
