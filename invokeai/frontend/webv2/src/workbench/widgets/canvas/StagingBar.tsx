/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { CanvasStagingAreaContractV2, CanvasStagingCandidateContract } from '@workbench/types';

import { HStack, Menu, Portal, ScrollArea, Spinner, Stack, Text } from '@chakra-ui/react';
import { Button, IconButton, MenuContent, toaster } from '@workbench/components/ui';
import { getImageThumbnailUrl, saveImageToGallery } from '@workbench/gallery/api';
import { CanvasFloatingBar } from '@workbench/widgets/canvas/CanvasFloatingBar';
import {
  CheckIcon,
  ChevronDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  ChevronUpIcon,
  EyeIcon,
  EyeOffIcon,
  ImagePlusIcon,
  SparklesIcon,
  Trash2Icon,
} from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

type AutoSwitchMode = CanvasStagingAreaContractV2['autoSwitchMode'];

const THUMBNAIL_STRIP_HEIGHT = '5rem';
const AUTO_SWITCH_MODES: AutoSwitchMode[] = ['off', 'latest', 'oldest'];
const MENU_POSITIONING = { placement: 'top-end' } as const;

interface StagingBarProps {
  areThumbnailsVisible: boolean;
  autoSwitchMode: AutoSwitchMode;
  hasMultipleCandidates: boolean;
  isGenerating: boolean;
  isVisible: boolean;
  pendingImages: CanvasStagingCandidateContract[];
  selectedCandidate: CanvasStagingCandidateContract | undefined;
  selectedImageIndex: number;
  onAccept: () => void;
  onCycle: (direction: -1 | 1) => void;
  onDiscardAll: () => void;
  onDiscardSelected: () => void;
  onSelectImage: (imageIndex: number) => void;
  onSetAutoSwitch: (mode: AutoSwitchMode) => void;
  onToggleThumbnails: () => void;
  onToggleVisibility: () => void;
}

/**
 * The floating staging bar over the canvas: appears while a canvas generation
 * is in flight or staged candidates await a decision. It drives the reducer's
 * staging actions (cycle / accept / discard / auto-switch); the candidate and
 * live progress pixels themselves are drawn on the engine canvas via
 * `engine.setStagedPreview` (wired in {@link CanvasWidgetView}). Rendered inside
 * the canvas's bottom-center floating group, stacked directly above the tool
 * options bar; positioning is the parent's job.
 */
export const StagingBar = ({
  areThumbnailsVisible,
  autoSwitchMode,
  hasMultipleCandidates,
  isGenerating,
  isVisible,
  pendingImages,
  selectedCandidate,
  selectedImageIndex,
  onAccept,
  onCycle,
  onDiscardAll,
  onDiscardSelected,
  onSelectImage,
  onSetAutoSwitch,
  onToggleThumbnails,
  onToggleVisibility,
}: StagingBarProps) => {
  const { t } = useTranslation();
  const [isSaving, setIsSaving] = useState(false);
  const hasCandidates = pendingImages.length > 0;

  const handleSaveToGallery = async () => {
    if (!selectedCandidate || isSaving) {
      return;
    }
    setIsSaving(true);
    try {
      await saveImageToGallery(selectedCandidate.imageName);
      toaster.create({
        description: t('widgets.canvas.staging.savedDescription', { name: selectedCandidate.imageName }),
        title: t('widgets.canvas.staging.saved'),
        type: 'success',
      });
    } catch {
      toaster.create({ title: t('widgets.canvas.staging.saveError'), type: 'error' });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Stack align="center" gap="2" w="full">
      {hasCandidates ? (
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
              <HStack h="full" justify="center">
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
      ) : null}

      <CanvasFloatingBar>
        <HStack gap="2">
          {isGenerating ? (
            <HStack color="fg.muted" gap="1.5" px="1">
              <Spinner size="xs" />
              <Text fontSize="xs" fontWeight="600">
                {t('widgets.canvas.staging.generating')}
              </Text>
            </HStack>
          ) : null}

          {hasCandidates && selectedCandidate ? (
            <>
              <IconButton
                aria-label={
                  areThumbnailsVisible
                    ? t('widgets.canvas.hideStagingThumbnails')
                    : t('widgets.canvas.showStagingThumbnails')
                }
                size="xs"
                variant="ghost"
                onClick={onToggleThumbnails}
              >
                {areThumbnailsVisible ? <ChevronDownIcon /> : <ChevronUpIcon />}
              </IconButton>

              <HStack gap="0.5">
                <IconButton
                  aria-label={t('widgets.canvas.previousStagedCandidate')}
                  disabled={!hasMultipleCandidates}
                  size="xs"
                  variant="ghost"
                  onClick={() => onCycle(-1)}
                >
                  <ChevronLeftIcon />
                </IconButton>
                <Text fontSize="xs" fontVariantNumeric="tabular-nums" minW="3.5rem" px="1" textAlign="center">
                  {t('widgets.canvas.candidateCount', {
                    current: selectedImageIndex + 1,
                    total: pendingImages.length,
                  })}
                </Text>
                <IconButton
                  aria-label={t('widgets.canvas.nextStagedCandidate')}
                  disabled={!hasMultipleCandidates}
                  size="xs"
                  variant="ghost"
                  onClick={() => onCycle(1)}
                >
                  <ChevronRightIcon />
                </IconButton>
              </HStack>

              <IconButton
                aria-label={
                  isVisible ? t('widgets.canvas.hideStagedResultPreview') : t('widgets.canvas.showStagedResultPreview')
                }
                size="xs"
                variant="ghost"
                onClick={onToggleVisibility}
              >
                {isVisible ? <EyeIcon /> : <EyeOffIcon />}
              </IconButton>

              <IconButton
                aria-label={t('widgets.canvas.staging.saveToGallery')}
                disabled={isSaving}
                size="xs"
                variant="ghost"
                onClick={handleSaveToGallery}
              >
                {isSaving ? <Spinner size="xs" /> : <ImagePlusIcon />}
              </IconButton>

              <IconButton aria-label={t('common.discard')} size="xs" variant="ghost" onClick={onDiscardSelected}>
                <Trash2Icon />
              </IconButton>

              <Button size="xs" variant="ghost" onClick={onDiscardAll}>
                {t('common.discardAll')}
              </Button>

              <AutoSwitchMenu mode={autoSwitchMode} onSelect={onSetAutoSwitch} />

              <Button size="xs" onClick={onAccept}>
                {t('widgets.canvas.acceptToLayer')}
              </Button>
            </>
          ) : null}
        </HStack>
      </CanvasFloatingBar>
    </Stack>
  );
};

const AutoSwitchMenu = ({ mode, onSelect }: { mode: AutoSwitchMode; onSelect: (mode: AutoSwitchMode) => void }) => {
  const { t } = useTranslation();
  const label = (value: AutoSwitchMode): string =>
    t(
      value === 'off'
        ? 'widgets.canvas.staging.autoSwitchOff'
        : value === 'latest'
          ? 'widgets.canvas.staging.autoSwitchLatest'
          : 'widgets.canvas.staging.autoSwitchOldest'
    );

  return (
    <Menu.Root positioning={MENU_POSITIONING}>
      <Menu.Trigger asChild>
        <IconButton aria-label={t('widgets.canvas.staging.autoSwitch')} minW="unset" px="2" size="xs" variant="ghost">
          <HStack gap="1">
            <SparklesIcon size={13} />
            <Text fontSize="xs">{label(mode)}</Text>
          </HStack>
        </IconButton>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="8rem" py="1">
            {AUTO_SWITCH_MODES.map((value) => (
              <Menu.Item key={value} value={value} onClick={() => onSelect(value)}>
                <CheckIcon size={12} opacity={mode === value ? 1 : 0} />
                <Menu.ItemText fontSize="xs">{label(value)}</Menu.ItemText>
              </Menu.Item>
            ))}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const StagingThumbnail = ({
  candidate,
  index,
  isSelected,
  onSelect,
}: {
  candidate: CanvasStagingCandidateContract;
  index: number;
  isSelected: boolean;
  onSelect: () => void;
}) => {
  const { t } = useTranslation();

  return (
    <Stack
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
        src={candidate.thumbnailUrl || getImageThumbnailUrl(candidate.imageName)}
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
    </Stack>
  );
};
