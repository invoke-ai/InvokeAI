import type { CanvasStagingCandidateContract, GeneratedImageContract } from '@workbench/types';

import { Box, HStack, ScrollArea, Stack, Text } from '@chakra-ui/react';
import { Button, IconButton } from '@workbench/components/ui';
import { ChevronDownIcon, ChevronLeftIcon, ChevronRightIcon, ChevronUpIcon, EyeIcon, EyeOffIcon } from 'lucide-react';

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
}: CanvasStagingControlsProps) => (
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
          Candidate {selectedImageIndex + 1} of {pendingImages.length}
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          {selectedCandidate.width} x {selectedCandidate.height}
        </Text>
      </HStack>
      <HStack justify="space-between">
        <HStack gap="1">
          <IconButton
            aria-label={areThumbnailsVisible ? 'Hide staging thumbnails' : 'Show staging thumbnails'}
            size="xs"
            variant="outline"
            onClick={onToggleThumbnails}
          >
            {areThumbnailsVisible ? <ChevronDownIcon /> : <ChevronUpIcon />}
          </IconButton>
          <IconButton
            aria-label="Previous staged candidate"
            disabled={!hasMultipleCandidates}
            size="xs"
            variant="outline"
            onClick={() => onCycle(-1)}
          >
            <ChevronLeftIcon />
          </IconButton>
          <IconButton
            aria-label="Next staged candidate"
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
            aria-label={isVisible ? 'Hide staged result preview' : 'Show staged result preview'}
            size="xs"
            variant="outline"
            onClick={onToggleVisibility}
          >
            {isVisible ? <EyeIcon /> : <EyeOffIcon />}
          </IconButton>
          <Button size="xs" variant="outline" onClick={onDiscardSelected}>
            Discard
          </Button>
          <Button size="xs" variant="outline" onClick={onDiscardAll}>
            Discard All
          </Button>
          <Button disabled={!isVisible} size="xs" onClick={onAccept}>
            Accept to Layer
          </Button>
        </HStack>
      </HStack>
    </Stack>
  </Stack>
);

export const EmptyStagingControls = () => (
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
      Invoke Generate to Canvas to stage results here.
    </Text>
  </Stack>
);

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
}) => (
  <Box
    aria-label={`Select staged candidate ${index + 1}`}
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
