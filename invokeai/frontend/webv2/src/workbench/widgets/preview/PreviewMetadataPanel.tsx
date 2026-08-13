import type { GalleryImage, GalleryImageMetadata, GalleryItem, GalleryItemKey } from '@features/gallery';

import { DataList, HStack, Icon, Stack, Tabs, Text } from '@chakra-ui/react';
import { galleryImages, galleryVideos } from '@features/gallery';
import { toGalleryItemKey } from '@features/gallery/contracts';
import { useAuthSession } from '@features/identity';
import { IconButton, Tooltip } from '@platform/ui';
import { JsonPreview } from '@platform/ui/JsonPreview';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { useQuery } from '@tanstack/react-query';
import {
  EMPTY_IMAGE_RECALL_CAPABILITIES,
  getImageRecallVerb,
  RecallActionButtons,
  type ImageActions,
  type ImageRecallCapabilities,
  type ImageRecallKind,
} from '@workbench/image-actions';
import { ChevronDownIcon, ChevronRightIcon, CopyIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { parsePreviewMetadata, type PreviewMetadataEntry } from './previewMetadata';

/**
 * The footer's collapsible "Details" section: how the image was made (prompt,
 * seed, model, sampler settings) as quiet label/value rows with per-row copy
 * and the recall verbs wired to the existing recall machinery. Video Details
 * expose the backend's raw metadata/workflow/graph payloads instead. The query
 * child exists only while expanded and is keyed by account + qualified item
 * identity, so closing or changing identity aborts supported transports.
 */

const GROUP_HOVER_VISIBLE = { opacity: 1 };

/**
 * Rows whose value maps onto a single-field recall verb. Rows without a
 * dedicated verb (model, steps, scheduler — only recallable via All/Remix)
 * keep just the copy button.
 */
const ENTRY_RECALL_KINDS: Partial<
  Record<string, { capability: keyof ImageRecallCapabilities; kind: ImageRecallKind }>
> = {
  clipSkip: { capability: 'clipSkip', kind: 'clipSkip' },
  negativePrompt: { capability: 'prompts', kind: 'prompts' },
  positivePrompt: { capability: 'prompts', kind: 'prompts' },
  seed: { capability: 'seed', kind: 'seed' },
  size: { capability: 'dimensions', kind: 'dimensions' },
};

export const PreviewMetadataPanel = ({
  actions,
  image,
  isOpen,
  item,
  onToggle,
}: {
  actions: ImageActions;
  image: GalleryImage | null;
  isOpen: boolean;
  item: GalleryItem;
  onToggle: () => void;
}) => {
  const { t } = useTranslation();
  const accountEpoch = useAuthSession().accountEpoch;
  const itemKey = toGalleryItemKey(item);

  return (
    <Stack gap="2">
      <HStack
        as="button"
        aria-expanded={isOpen}
        color="fg.muted"
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
        <PreviewDetailsQuery
          key={`${accountEpoch}:${itemKey}`}
          accountEpoch={accountEpoch}
          actions={actions}
          image={image}
          item={item}
          itemKey={itemKey}
        />
      ) : null}
    </Stack>
  );
};

type PreviewDetailsData =
  | {
      kind: 'image';
      metadata: GalleryImageMetadata | null;
    }
  | {
      graph: string | null;
      kind: 'video';
      metadata: Record<string, unknown> | null;
      workflow: string | null;
    };

const PreviewDetailsQuery = ({
  accountEpoch,
  actions,
  image,
  item,
  itemKey,
}: {
  accountEpoch: number;
  actions: ImageActions;
  image: GalleryImage | null;
  item: GalleryItem;
  itemKey: GalleryItemKey;
}) => {
  const { t } = useTranslation();
  const detailsQuery = useQuery({
    queryFn: async ({ signal }): Promise<PreviewDetailsData> => {
      if (item.kind === 'image') {
        if (!image) {
          return { kind: 'image', metadata: null };
        }

        const metadata = await galleryImages.metadata(item.name, signal);

        return { kind: 'image', metadata };
      }

      const [metadata, workflow] = await Promise.all([
        galleryVideos.metadata(item.name, signal),
        galleryVideos.workflow(item.name, signal),
      ]);

      return { graph: workflow.graph, kind: 'video', metadata, workflow: workflow.workflow };
    },
    queryKey: ['preview', 'details', accountEpoch, itemKey],
    retry: false,
  });

  if (item.kind === 'video') {
    if (detailsQuery.isPending) {
      return (
        <Text color="fg.subtle" fontSize="2xs">
          {t('widgets.preview.loadingMetadata')}
        </Text>
      );
    }

    const details = isVideoDetails(detailsQuery.data) ? detailsQuery.data : null;

    return (
      <VideoDetails
        graph={details?.graph ?? null}
        metadata={details?.metadata ?? null}
        workflow={details?.workflow ?? null}
      />
    );
  }

  const details = isImageDetails(detailsQuery.data) ? detailsQuery.data : null;
  const capabilities =
    image && details ? actions.deriveImageRecallCapabilities(image, details.metadata) : EMPTY_IMAGE_RECALL_CAPABILITIES;

  return (
    <ImageDetails
      actions={actions}
      capabilities={capabilities}
      image={image}
      isLoading={detailsQuery.isPending}
      metadata={details?.metadata ?? null}
    />
  );
};

const isVideoDetails = (
  value: PreviewDetailsData | undefined
): value is Extract<PreviewDetailsData, { kind: 'video' }> => value?.kind === 'video';

const isImageDetails = (
  value: PreviewDetailsData | undefined
): value is Extract<PreviewDetailsData, { kind: 'image' }> => value?.kind === 'image';

const ImageDetails = ({
  actions,
  capabilities,
  image,
  isLoading,
  metadata,
}: {
  actions: ImageActions;
  capabilities: ImageRecallCapabilities;
  image: GalleryImage | null;
  isLoading: boolean;
  metadata: GalleryImageMetadata | null;
}) => {
  const { t } = useTranslation();
  const entries = [
    ...parsePreviewMetadata(metadata),
    ...(image ? [{ key: 'sourceRun', label: 'Source Run', value: image.sourceQueueItemId }] : []),
  ];
  const handleRecall = useCallback(
    (kind: ImageRecallKind) => {
      if (image) {
        void actions.recallImageData(image, kind);
      }
    },
    [actions, image]
  );

  return (
    <Stack gap="2" maxH="40cqh" overflowY="auto" pe="1">
      {isLoading ? (
        <Text color="fg.subtle" fontSize="2xs">
          {t('widgets.preview.loadingMetadata')}
        </Text>
      ) : (
        <DataList.Root gap="1.5" orientation="horizontal" size="sm">
          {entries.map((entry) => {
            const recall = ENTRY_RECALL_KINDS[entry.key];

            return (
              <MetadataRow
                key={entry.key}
                entry={entry}
                recallKind={recall && capabilities[recall.capability] ? recall.kind : undefined}
                onRecall={handleRecall}
              />
            );
          })}
        </DataList.Root>
      )}
      <RecallActionButtons
        capabilities={capabilities}
        disabledReason={t('widgets.preview.recallNotAvailable')}
        onRecall={handleRecall}
      />
    </Stack>
  );
};

const VideoDetails = ({
  graph,
  metadata,
  workflow,
}: {
  graph: string | null;
  metadata: Record<string, unknown> | null;
  workflow: string | null;
}) => {
  const { t } = useTranslation();

  return (
    <Tabs.Root defaultValue="metadata" lazyMount size="sm" unmountOnExit variant="outline">
      <Tabs.List>
        <Tabs.Trigger fontSize="2xs" value="metadata">
          {t('widgets.preview.metadata')}
        </Tabs.Trigger>
        <Tabs.Trigger fontSize="2xs" value="workflow">
          {t('widgets.preview.workflow')}
        </Tabs.Trigger>
        <Tabs.Trigger fontSize="2xs" value="graph">
          {t('widgets.preview.graph')}
        </Tabs.Trigger>
      </Tabs.List>
      <Tabs.Content value="metadata">
        <JsonPreview label={t('widgets.preview.metadataJsonLabel')} maxH="40cqh" value={metadata} />
      </Tabs.Content>
      <Tabs.Content value="workflow">
        <RawJsonPreview label={t('widgets.preview.workflowJsonLabel')} text={workflow} />
      </Tabs.Content>
      <Tabs.Content value="graph">
        <RawJsonPreview label={t('widgets.preview.graphJsonLabel')} text={graph} />
      </Tabs.Content>
    </Tabs.Root>
  );
};

const RawJsonPreview = ({ label, text }: { label: string; text: string | null }) =>
  text === null ? (
    <JsonPreview label={label} maxH="40cqh" value={null} />
  ) : (
    <JsonPreview label={label} maxH="40cqh" text={text} />
  );

/**
 * A DataList item extended with hover-revealed value actions: recall (when
 * the row maps onto a single-field recall verb) and copy. The recall button
 * reuses the verb row's icon and label so both affordances read as the same
 * action.
 */
const MetadataRow = ({
  entry,
  onRecall,
  recallKind,
}: {
  entry: PreviewMetadataEntry;
  onRecall: (kind: ImageRecallKind) => void;
  recallKind?: ImageRecallKind;
}) => {
  const { t } = useTranslation();
  const copyValue = useCallback(() => void navigator.clipboard.writeText(entry.value), [entry.value]);
  const recallValue = useCallback(() => {
    if (recallKind) {
      onRecall(recallKind);
    }
  }, [onRecall, recallKind]);
  const recallVerb = recallKind ? getImageRecallVerb(recallKind) : null;

  return (
    <DataList.Item alignItems="start" className="group">
      <DataList.ItemLabel fontSize="2xs">{entry.label}</DataList.ItemLabel>
      <DataList.ItemValue fontSize="2xs" minW="0">
        {entry.isMultiline ? (
          <Text flex="1" fontSize="2xs" minW="0" whiteSpace="pre-wrap">
            {entry.value}
          </Text>
        ) : (
          <MiddleTruncate flex="1" fontSize="2xs" minW="0" text={entry.value} />
        )}
        <HStack
          flexShrink={0}
          gap="0"
          opacity={0}
          transitionDuration="var(--wb-motion-duration-fast)"
          transitionProperty="opacity"
          _groupHover={GROUP_HOVER_VISIBLE}
        >
          {recallVerb ? (
            <Tooltip content={recallVerb.label}>
              <IconButton
                aria-label={recallVerb.label}
                color="fg.muted"
                size="2xs"
                variant="ghost"
                onClick={recallValue}
              >
                <Icon as={recallVerb.icon} boxSize="3" />
              </IconButton>
            </Tooltip>
          ) : null}
          <Tooltip content={t('common.copy')}>
            <IconButton aria-label={t('common.copy')} color="fg.muted" size="2xs" variant="ghost" onClick={copyValue}>
              <Icon as={CopyIcon} boxSize="3" />
            </IconButton>
          </Tooltip>
        </HStack>
      </DataList.ItemValue>
    </DataList.Item>
  );
};
