import type { GalleryImage, GalleryImageMetadata, GalleryItem, GalleryItemKey } from '@features/gallery';

import { DataList, HStack, Icon, Stack, Tabs, Text } from '@chakra-ui/react';
import { galleryImages, galleryVideos } from '@features/gallery';
import { toGalleryItemKey } from '@features/gallery/contracts';
import { IconButton, Scrollable, Tooltip } from '@platform/ui';
import { JsonPreview } from '@platform/ui/JsonPreview';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { useQuery } from '@tanstack/react-query';
import {
  EMPTY_IMAGE_RECALL_CAPABILITIES,
  getImageRecallVerb,
  type ImageActions,
  type ImageRecallCapabilities,
  type ImageRecallKind,
} from '@workbench/image-actions';
import { CopyIcon } from 'lucide-react';
import { useCallback, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import { parsePreviewMetadata, type PreviewMetadataEntry } from './previewMetadata';

/**
 * Show parsed image details with copy/recall, or raw video payloads. Key query children by account and qualified
 * identity so closure/identity changes abort supported transports.
 */

const GROUP_HOVER_VISIBLE = { opacity: 1 };

/**
 * Offer field recall only where a dedicated verb exists; model/steps/scheduler retain copy and use All/Remix for
 * recall.
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

export const PreviewDetails = ({
  accountEpoch,
  actions,
  image,
  item,
}: {
  /** Read by the caller, outside the popover's presence boundary, so an account change re-keys the query. */
  accountEpoch: number;
  actions: ImageActions;
  image: GalleryImage | null;
  item: GalleryItem;
}) => {
  const itemKey = toGalleryItemKey(item);

  return (
    <PreviewDetailsQuery
      key={`${accountEpoch}:${itemKey}`}
      accountEpoch={accountEpoch}
      actions={actions}
      image={image}
      item={item}
      itemKey={itemKey}
    />
  );
};

type PreviewDetailsData =
  | {
      graph: string | null;
      kind: 'image';
      metadata: GalleryImageMetadata | null;
      workflow: string | null;
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
          return { graph: null, kind: 'image', metadata: null, workflow: null };
        }

        // Show complete raw metadata alongside parsed fields because workflows may add arbitrary entries.
        const [metadata, workflow] = await Promise.all([
          galleryImages.metadata(item.name, signal),
          galleryImages.workflow(item.name, signal),
        ]);

        return { graph: workflow.graph, kind: 'image', metadata, workflow: workflow.workflow };
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
      <DetailsTabs
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
    <DetailsTabs
      graph={details?.graph ?? null}
      metadata={details?.metadata ?? null}
      workflow={details?.workflow ?? null}
    >
      <ImageDetails
        actions={actions}
        capabilities={capabilities}
        image={image}
        isLoading={detailsQuery.isPending}
        metadata={details?.metadata ?? null}
      />
    </DetailsTabs>
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
    <Scrollable flex="1" minH="0">
      <Stack gap="2" pe="1">
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
      </Stack>
    </Scrollable>
  );
};

const DetailsTabs = ({
  children: details,
  graph,
  metadata,
  workflow,
}: {
  /** The parsed rows + recall verbs; images only. */
  children?: ReactNode;
  graph: string | null;
  metadata: Record<string, unknown> | null;
  workflow: string | null;
}) => {
  const { t } = useTranslation();
  // A tab with nothing behind it is disabled rather than opening onto an empty
  // pane, so the first tab with content is where the popover opens.
  const defaultValue = details ? 'details' : metadata !== null ? 'metadata' : workflow !== null ? 'workflow' : 'graph';

  return (
    <Tabs.Root
      defaultValue={defaultValue}
      display="flex"
      flexDirection="column"
      lazyMount
      minH="0"
      size="sm"
      unmountOnExit
      variant="outline"
    >
      <Tabs.List flexShrink={0}>
        {details ? (
          <Tabs.Trigger fontSize="2xs" value="details">
            {t('widgets.preview.details')}
          </Tabs.Trigger>
        ) : null}
        <Tabs.Trigger disabled={metadata === null} fontSize="2xs" value="metadata">
          {t('widgets.preview.metadata')}
        </Tabs.Trigger>
        <Tabs.Trigger disabled={workflow === null} fontSize="2xs" value="workflow">
          {t('widgets.preview.workflow')}
        </Tabs.Trigger>
        <Tabs.Trigger disabled={graph === null} fontSize="2xs" value="graph">
          {t('widgets.preview.graph')}
        </Tabs.Trigger>
      </Tabs.List>
      {details ? (
        <Tabs.Content display="flex" flexDirection="column" minH="0" value="details">
          {details}
        </Tabs.Content>
      ) : null}
      <Tabs.Content display="flex" flexDirection="column" minH="0" value="metadata">
        <JsonPreview label={t('widgets.preview.metadataJsonLabel')} maxH="100%" value={metadata} />
      </Tabs.Content>
      <Tabs.Content display="flex" flexDirection="column" minH="0" value="workflow">
        <RawJsonPreview label={t('widgets.preview.workflowJsonLabel')} text={workflow} />
      </Tabs.Content>
      <Tabs.Content display="flex" flexDirection="column" minH="0" value="graph">
        <RawJsonPreview label={t('widgets.preview.graphJsonLabel')} text={graph} />
      </Tabs.Content>
    </Tabs.Root>
  );
};

const RawJsonPreview = ({ label, text }: { label: string; text: string | null }) =>
  text === null ? (
    <JsonPreview label={label} maxH="100%" value={null} />
  ) : (
    <JsonPreview label={label} maxH="100%" text={text} />
  );

/** Share recall labels/icons with the verb row; reveal row-level recall and copy actions on hover. */
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
      <DataList.ItemValue alignItems="flex-start" fontSize="2xs" minW="0">
        {entry.isMultiline ? (
          <Text flex="1" fontSize="2xs" minW="0" whiteSpace="pre-wrap">
            {entry.value}
          </Text>
        ) : (
          <MiddleTruncate flex="1" fontSize="2xs" minW="0" text={entry.value} />
        )}
        <HStack
          alignSelf="flex-start"
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
