import type { GalleryImage, GalleryImageMetadata } from '@workbench/gallery/api';

import { Box, DataList, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { Button, IconButton, Tooltip } from '@workbench/components/ui';
import { getGalleryImageMetadata } from '@workbench/gallery/api';
import {
  EMPTY_IMAGE_RECALL_CAPABILITIES,
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

const RECALL_ITEMS: { capability: keyof ImageRecallCapabilities; kind: ImageRecallKind; label: string }[] = [
  { capability: 'all', kind: 'all', label: 'Recall All' },
  { capability: 'remix', kind: 'remix', label: 'Remix' },
  { capability: 'prompts', kind: 'prompts', label: 'Use Prompt' },
  { capability: 'seed', kind: 'seed', label: 'Use Seed' },
  { capability: 'dimensions', kind: 'dimensions', label: 'Use Size' },
  { capability: 'clipSkip', kind: 'clipSkip', label: 'Use CLIP Skip' },
];

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
    metadata: GalleryImageMetadata | null;
  } | null>(null);
  const isLoading = isOpen && loaded === null;

  // Metadata is immutable per image and this component is keyed per image, so
  // fetch exactly once, on first expand.
  useEffect(() => {
    if (!isOpen || loaded !== null) {
      return;
    }

    let isStale = false;

    Promise.all([getGalleryImageMetadata(image.imageName), actions.getImageRecallCapabilities(image)])
      .then(([metadata, capabilities]) => {
        if (!isStale) {
          setLoaded({ capabilities, metadata });
        }
      })
      .catch(() => {
        if (!isStale) {
          setLoaded({ capabilities: EMPTY_IMAGE_RECALL_CAPABILITIES, metadata: null });
        }
      });

    return () => {
      isStale = true;
    };
  }, [actions, image, isOpen, loaded]);

  const entries = parsePreviewMetadata(loaded?.metadata ?? null);
  const recallItems = loaded ? RECALL_ITEMS.filter((item) => loaded.capabilities[item.capability]) : [];

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
        <Stack gap="2" maxH="40cqh" overflowY="auto" pe="1">
          {isLoading ? (
            <Text color="fg.subtle" fontSize="2xs">
              {t('widgets.preview.loadingMetadata')}
            </Text>
          ) : entries.length === 0 ? (
            <Text color="fg.subtle" fontSize="2xs">
              {t('widgets.preview.noMetadata')}
            </Text>
          ) : (
            <>
              <DataList.Root gap="1.5" orientation="horizontal" size="sm">
                {entries.map((entry) => (
                  <MetadataRow key={entry.key} entry={entry} />
                ))}
              </DataList.Root>
              {recallItems.length > 0 ? (
                <HStack flexWrap="wrap" gap="1">
                  {recallItems.map((item) => (
                    <RecallButton key={item.kind} actions={actions} image={image} kind={item.kind} label={item.label} />
                  ))}
                </HStack>
              ) : null}
            </>
          )}
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

const RecallButton = ({
  actions,
  image,
  kind,
  label,
}: {
  actions: ImageActions;
  image: GalleryImage;
  kind: ImageRecallKind;
  label: string;
}) => {
  const handleClick = useCallback(() => void actions.recallImageData(image, kind), [actions, image, kind]);

  return (
    <Button size="2xs" variant="outline" onClick={handleClick}>
      {label}
    </Button>
  );
};
