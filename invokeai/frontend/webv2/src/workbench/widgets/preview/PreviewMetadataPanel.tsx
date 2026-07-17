import type { GalleryImage, GalleryImageMetadata } from '@workbench/gallery/api';

import { Box, DataList, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@workbench/components/ui';
import { getGalleryImageMetadata } from '@workbench/gallery/api';
import {
  EMPTY_IMAGE_RECALL_CAPABILITIES,
  RecallActionButtons,
  type ImageActions,
  type ImageRecallCapabilities,
  type ImageRecallKind,
} from '@workbench/image-actions';
import { ChevronDownIcon, ChevronRightIcon, CopyIcon } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { parsePreviewMetadata, type PreviewMetadataEntry } from './previewMetadata';

/**
 * The footer's collapsible "Details" section: how the image was made (prompt,
 * seed, model, sampler settings) as quiet label/value rows with per-row copy
 * and the recall verbs wired to the existing recall machinery. Metadata is
 * fetched only while expanded; the parent keys this component per image so
 * stale rows never survive a selection change.
 */

const GROUP_HOVER_VISIBLE = { opacity: 1 };

export const PreviewMetadataPanel = ({
  actions,
  image,
  isOpen,
  onToggle,
}: {
  actions: ImageActions;
  image: GalleryImage;
  isOpen: boolean;
  onToggle: () => void;
}) => {
  const { t } = useTranslation();
  const [loaded, setLoaded] = useState<{
    capabilities: ImageRecallCapabilities;
    imageName: string;
    metadata: GalleryImageMetadata | null;
  } | null>(null);
  // Stale-while-loading: when the selection changes, the previous image's rows
  // stay in place (slightly dimmed) until the new metadata arrives, so
  // navigating never collapses the panel or shifts the layout.
  const isCurrent = loaded?.imageName === image.imageName;
  const isLoading = isOpen && loaded === null;

  useEffect(() => {
    if (!isOpen || loaded?.imageName === image.imageName) {
      return;
    }

    let isStale = false;

    Promise.all([getGalleryImageMetadata(image.imageName), actions.getImageRecallCapabilities(image)])
      .then(([metadata, capabilities]) => {
        if (!isStale) {
          setLoaded({ capabilities, imageName: image.imageName, metadata });
        }
      })
      .catch(() => {
        if (!isStale) {
          setLoaded({ capabilities: EMPTY_IMAGE_RECALL_CAPABILITIES, imageName: image.imageName, metadata: null });
        }
      });

    return () => {
      isStale = true;
    };
  }, [actions, image, isOpen, loaded?.imageName]);

  // Source run is identity the status bar no longer shows; it reads as one
  // more metadata fact here.
  const entries = [
    ...parsePreviewMetadata(loaded?.metadata ?? null),
    { key: 'sourceRun', label: 'Source Run', value: image.sourceQueueItemId },
  ];
  const handleRecall = useCallback(
    (kind: ImageRecallKind) => void actions.recallImageData(image, kind),
    [actions, image]
  );
  // The recall row is part of the panel's fixed skeleton — never unmounted on
  // navigation, so it can't cause layout shift. While the next image's
  // capabilities load, the previous ones stay displayed (the panel dims to
  // signal staleness); clicks remain safe because `recallImageData` targets
  // the CURRENT image and re-validates against its fresh metadata. Only the
  // first-ever load shows all verbs disabled.
  const capabilities = loaded?.capabilities ?? EMPTY_IMAGE_RECALL_CAPABILITIES;

  return (
    <Stack gap="2">
      <HStack
        as="button"
        aria-expanded={isOpen}
        color="fg.subtle"
        cursor="pointer"
        gap="1"
        w="fit-content"
        onClick={onToggle}
      >
        <Icon as={isOpen ? ChevronDownIcon : ChevronRightIcon} boxSize="3" />
        <Text fontSize="2xs" fontWeight="700" textTransform="uppercase">
          {t('widgets.preview.details')}
        </Text>
      </HStack>
      {isOpen ? (
        <Stack
          gap="2"
          maxH="40cqh"
          opacity={isCurrent || loaded === null ? 1 : 0.6}
          overflowY="auto"
          pe="1"
          transitionDuration="var(--wb-motion-duration-fast)"
          transitionProperty="opacity"
        >
          {isLoading ? (
            <Text color="fg.subtle" fontSize="2xs">
              {t('widgets.preview.loadingMetadata')}
            </Text>
          ) : (
            <DataList.Root gap="1.5" orientation="horizontal" size="sm">
              {entries.map((entry) => (
                <MetadataRow key={entry.key} entry={entry} />
              ))}
            </DataList.Root>
          )}
          <RecallActionButtons
            capabilities={capabilities}
            disabledReason={t('widgets.preview.recallNotAvailable')}
            onRecall={handleRecall}
          />
        </Stack>
      ) : null}
    </Stack>
  );
};

/** A DataList item extended with a hover-revealed copy button on the value. */
const MetadataRow = ({ entry }: { entry: PreviewMetadataEntry }) => {
  const { t } = useTranslation();
  const copyValue = useCallback(() => void navigator.clipboard.writeText(entry.value), [entry.value]);

  return (
    <DataList.Item alignItems="start" className="group">
      <DataList.ItemLabel fontSize="2xs">{entry.label}</DataList.ItemLabel>
      <DataList.ItemValue fontSize="2xs" minW="0">
        <Text
          flex="1"
          fontSize="2xs"
          minW="0"
          whiteSpace={entry.isMultiline ? 'pre-wrap' : undefined}
          truncate={!entry.isMultiline}
        >
          {entry.value}
        </Text>
        <Box
          flexShrink={0}
          opacity={0}
          transitionDuration="var(--wb-motion-duration-fast)"
          transitionProperty="opacity"
          _groupHover={GROUP_HOVER_VISIBLE}
        >
          <Tooltip content={t('common.copy')}>
            <IconButton aria-label={t('common.copy')} color="fg.muted" size="2xs" variant="ghost" onClick={copyValue}>
              <Icon as={CopyIcon} boxSize="3" />
            </IconButton>
          </Tooltip>
        </Box>
      </DataList.ItemValue>
    </DataList.Item>
  );
};
