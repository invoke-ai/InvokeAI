import type { GalleryImage, GalleryImageMetadata, GalleryItem, GalleryItemKey } from '@features/gallery';

import { Box, DataList, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { galleryImages, galleryVideos } from '@features/gallery';
import { toGalleryItemKey } from '@features/gallery/contracts';
import { canCopyText, copyText } from '@platform/browser/clipboard';
import { IconButton, Scrollable, Tooltip } from '@platform/ui';
import { JsonPreview } from '@platform/ui/JsonPreview';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { SegmentTabs, segmentTabsPanelId, segmentTabsTabId } from '@platform/ui/SegmentTabs';
import { toaster } from '@platform/ui/toaster';
import { useQuery } from '@tanstack/react-query';
import {
  EMPTY_IMAGE_RECALL_CAPABILITIES,
  getImageRecallVerb,
  type ImageActions,
  type ImageRecallCapabilities,
  type ImageRecallKind,
} from '@workbench/image-actions';
import { CopyIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useState, type KeyboardEvent, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import { parsePreviewMetadata, type PreviewMetadataEntry } from './previewMetadata';

/**
 * Show parsed image details with copy/recall, or raw video payloads. Key query children by account and qualified
 * identity so closure/identity changes abort supported transports.
 */

const GROUP_HOVER_VISIBLE = { opacity: 1 };

const TAB_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'End', 'Home']);

/** The tab list moves between its tabs with these keys; Preview's arrow bindings must not step the item too. */
const keepTabKeys = (event: KeyboardEvent<HTMLElement>) => {
  if (TAB_KEYS.has(event.key)) {
    event.stopPropagation();
  }
};

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
  /** Read by the caller, outside the panel's mount boundary, so an account change re-keys the query. */
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
        <Text color="fg.subtle" fontSize="xs">
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
          <Text color="fg.subtle" fontSize="xs">
            {t('widgets.preview.loadingMetadata')}
          </Text>
        ) : (
          <DataList.Root gap="1.5" orientation="horizontal">
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

type DetailsTabId = 'details' | 'graph' | 'metadata' | 'workflow';

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
  const idBase = useId();
  // A tab with nothing behind it is disabled rather than opening onto an empty
  // pane, so the first tab with content is where the panel opens.
  const [activeId, setActiveId] = useState<DetailsTabId>(() =>
    details ? 'details' : metadata !== null ? 'metadata' : workflow !== null ? 'workflow' : 'graph'
  );
  const tabs = useMemo(
    () => [
      ...(details ? [{ id: 'details' as const, label: t('widgets.preview.details') }] : []),
      { disabled: metadata === null, id: 'metadata' as const, label: t('widgets.preview.metadata') },
      { disabled: workflow === null, id: 'workflow' as const, label: t('widgets.preview.workflow') },
      { disabled: graph === null, id: 'graph' as const, label: t('widgets.preview.graph') },
    ],
    [details, graph, metadata, t, workflow]
  );

  return (
    <Stack gap="1.5" minH="0">
      <Box flexShrink={0} onKeyDown={keepTabKeys}>
        <SegmentTabs
          activeId={activeId}
          ariaLabel={t('widgets.preview.details')}
          idBase={idBase}
          isCompact
          tabs={tabs}
          onSelect={setActiveId}
        />
      </Box>
      <Stack
        aria-labelledby={segmentTabsTabId(idBase, activeId)}
        gap="0"
        id={segmentTabsPanelId(idBase)}
        minH="0"
        role="tabpanel"
      >
        {activeId === 'details' ? (
          details
        ) : activeId === 'metadata' ? (
          <JsonPreview label={t('widgets.preview.metadataJsonLabel')} maxH="100%" value={metadata} />
        ) : activeId === 'workflow' ? (
          <RawJsonPreview label={t('widgets.preview.workflowJsonLabel')} text={workflow} />
        ) : (
          <RawJsonPreview label={t('widgets.preview.graphJsonLabel')} text={graph} />
        )}
      </Stack>
    </Stack>
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
  const canCopy = canCopyText();
  const copyValue = useCallback(
    () =>
      void copyText(entry.value).catch(() =>
        toaster.create({ description: t('common.clipboardBlocked'), title: t('common.couldNotCopy'), type: 'error' })
      ),
    [entry.value, t]
  );
  const recallValue = useCallback(() => {
    if (recallKind) {
      onRecall(recallKind);
    }
  }, [onRecall, recallKind]);
  const recallVerb = recallKind ? getImageRecallVerb(recallKind) : null;

  return (
    <DataList.Item alignItems="start" className="group">
      <DataList.ItemLabel fontSize="xs">{entry.label}</DataList.ItemLabel>
      <DataList.ItemValue alignItems="flex-start" fontSize="xs" minW="0">
        {entry.isMultiline ? (
          <Text flex="1" fontSize="xs" minW="0" whiteSpace="pre-wrap">
            {entry.value}
          </Text>
        ) : (
          <MiddleTruncate flex="1" fontSize="xs" minW="0" text={entry.value} />
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
                size="sm"
                variant="ghost"
                onClick={recallValue}
              >
                <Icon as={recallVerb.icon} boxSize="3" />
              </IconButton>
            </Tooltip>
          ) : null}
          <IconButton
            tooltip={t('common.copy')}
            unavailableReason={canCopy ? undefined : t('common.copyUnavailable')}
            aria-label={t('common.copy')}
            color="fg.muted"
            size="sm"
            variant="ghost"
            onClick={copyValue}
          >
            <Icon as={CopyIcon} boxSize="3" />
          </IconButton>
        </HStack>
      </DataList.ItemValue>
    </DataList.Item>
  );
};
