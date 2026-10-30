import type { GalleryImage, GalleryImageMetadata, GalleryItem, GalleryItemKey } from '@features/gallery';

import { Box, DataList, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { galleryImages, galleryVideos } from '@features/gallery';
import { toGalleryItemKey } from '@features/gallery/contracts';
import { galleryImageMetadataOptions, galleryVideoMetadataOptions } from '@features/gallery/queries';
import { canCopyText, copyText } from '@platform/browser/clipboard';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { IconButton, Scrollable, Tooltip } from '@platform/ui';
import { JsonPreview } from '@platform/ui/JsonPreview';
import { LoadError } from '@platform/ui/LoadError';
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
import { useCallback, useId, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import { parsePreviewMetadata, type PreviewMetadataEntry } from './previewMetadata';

/**
 * Show parsed details with copy (and per-field recall for images) beside the raw payloads. Key query children by
 * account and qualified identity so closure/identity changes abort supported transports.
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
}) =>
  item.kind === 'video' ? (
    <VideoDetailsQuery accountEpoch={accountEpoch} itemKey={itemKey} videoName={item.name} />
  ) : image ? (
    <ImageDetailsQuery accountEpoch={accountEpoch} actions={actions} image={image} itemKey={itemKey} />
  ) : (
    <DetailsTabs graph={null} metadata={null} workflow={null}>
      <ImageDetails
        actions={actions}
        capabilities={EMPTY_IMAGE_RECALL_CAPABILITIES}
        image={null}
        isLoading={false}
        metadata={null}
      />
    </DetailsTabs>
  );

/**
 * Metadata comes from the gallery's shared entry, which the video's menu and recall commands also read; the embedded
 * workflow and graph are Details' own read. Either failing leaves the other's tabs on offer.
 */
const VideoDetailsQuery = ({
  accountEpoch,
  itemKey,
  videoName,
}: {
  accountEpoch: number;
  itemKey: GalleryItemKey;
  videoName: string;
}) => {
  const { t } = useTranslation();
  const metadataQuery = useQuery(galleryVideoMetadataOptions(captureAccountScope(), videoName));
  const workflowQuery = useQuery({
    queryFn: ({ signal }) => galleryVideos.workflow(videoName, signal),
    queryKey: ['preview', 'details', accountEpoch, itemKey, 'workflow'],
    retry: false,
  });

  if (metadataQuery.isPending || workflowQuery.isPending) {
    return (
      <Text color="fg.subtle" fontSize="xs">
        {t('widgets.preview.loadingMetadata')}
      </Text>
    );
  }

  const metadata = metadataQuery.data ?? null;
  // Recall verbs live in the video's menu; these rows only show and copy what was recorded.
  const entries = parsePreviewMetadata(metadata, { kind: 'video', t });

  return (
    <DetailsTabs
      failedReads={getFailedReads(metadataQuery, workflowQuery)}
      graph={workflowQuery.data?.graph ?? null}
      metadata={metadata}
      workflow={workflowQuery.data?.workflow ?? null}
    >
      {entries.length > 0 ? <VideoDetails entries={entries} /> : null}
    </DetailsTabs>
  );
};

const VideoDetails = ({ entries }: { entries: readonly PreviewMetadataEntry[] }) => (
  <Scrollable flex="1" minH="0">
    <Stack gap="2" pe="1">
      <DataList.Root gap="1.5" orientation="horizontal">
        {entries.map((entry) => (
          <MetadataRow key={entry.key} entry={entry} />
        ))}
      </DataList.Root>
    </Stack>
  </Scrollable>
);

/**
 * Metadata comes from the gallery's shared entry, which the image's menus and recall commands also read; the embedded
 * workflow and graph are Details' own read. Either failing leaves the other's tabs on offer.
 */
const ImageDetailsQuery = ({
  accountEpoch,
  actions,
  image,
  itemKey,
}: {
  accountEpoch: number;
  actions: ImageActions;
  image: GalleryImage;
  itemKey: GalleryItemKey;
}) => {
  const metadataQuery = useQuery(galleryImageMetadataOptions(captureAccountScope(), image.imageName));
  const workflowQuery = useQuery({
    queryFn: ({ signal }) => galleryImages.workflow(image.imageName, signal),
    queryKey: ['preview', 'details', accountEpoch, itemKey, 'workflow'],
    retry: false,
  });
  // Show complete raw metadata alongside parsed fields because workflows may add arbitrary entries.
  const metadata = metadataQuery.data ?? null;
  const capabilities = metadataQuery.isPending
    ? EMPTY_IMAGE_RECALL_CAPABILITIES
    : actions.deriveImageRecallCapabilities(image, metadata);

  return (
    <DetailsTabs
      failedReads={getFailedReads(metadataQuery, workflowQuery)}
      graph={workflowQuery.data?.resolved_inputs ?? workflowQuery.data?.graph ?? null}
      metadata={metadata}
      workflow={workflowQuery.data?.workflow ?? null}
    >
      <ImageDetails
        actions={actions}
        capabilities={capabilities}
        image={image}
        isLoading={metadataQuery.isPending}
        metadata={metadata}
      />
    </DetailsTabs>
  );
};

/** A Details read that failed: it holds no answer, unlike a null one (nothing recorded), so it offers a retry. */
interface FailedDetailsRead {
  isRetrying: boolean;
  read: 'metadata' | 'workflow';
  retry: () => unknown;
}

interface DetailsReadState {
  isError: boolean;
  isFetching: boolean;
  refetch: () => Promise<unknown>;
}

const getFailedReads = (metadata: DetailsReadState, workflow: DetailsReadState): FailedDetailsRead[] =>
  (
    [
      ['metadata', metadata],
      ['workflow', workflow],
    ] as const
  ).flatMap(([read, query]) =>
    query.isError ? [{ isRetrying: query.isFetching, read, retry: () => query.refetch() }] : []
  );

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
    ...parsePreviewMetadata(metadata, { kind: 'image', t }),
    ...(image
      ? [{ key: 'sourceRun', label: t('widgets.preview.fields.sourceRun'), value: image.sourceQueueItemId }]
      : []),
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

const NO_FAILED_READS: readonly FailedDetailsRead[] = [];

const DetailsTabs = ({
  children: details,
  failedReads = NO_FAILED_READS,
  graph,
  metadata,
  workflow,
}: {
  /** The parsed rows; the Details tab is offered only with them. */
  children?: ReactNode;
  failedReads?: readonly FailedDetailsRead[];
  graph: string | null;
  metadata: Record<string, unknown> | null;
  workflow: string | null;
}) => {
  const { t } = useTranslation();
  const idBase = useId();
  const tabListRef = useRef<HTMLDivElement | null>(null);
  // A tab with nothing behind it is disabled rather than opening onto an empty pane, so until a tab is chosen the panel
  // shows the first one with content, which a retried read can change. With none, Metadata stays selected over no pane.
  const [chosenId, setChosenId] = useState<DetailsTabId | null>(null);
  // A successful retry removes its button; focus goes to the selected tab rather than falling to the document.
  const focusSelectedTab = useCallback(() => {
    tabListRef.current?.querySelector<HTMLElement>('[role="tab"][aria-selected="true"]')?.focus();
  }, []);
  const tabs = useMemo(
    () => [
      ...(details ? [{ disabled: false, id: 'details' as const, label: t('widgets.preview.details') }] : []),
      { disabled: metadata === null, id: 'metadata' as const, label: t('widgets.preview.metadata') },
      { disabled: workflow === null, id: 'workflow' as const, label: t('widgets.preview.workflow') },
      { disabled: graph === null, id: 'graph' as const, label: t('widgets.preview.graph') },
    ],
    [details, graph, metadata, t, workflow]
  );
  const activeId = chosenId ?? tabs.find((tab) => !tab.disabled)?.id ?? 'metadata';
  const hasActiveContent = tabs.some((tab) => tab.id === activeId && !tab.disabled);

  return (
    <Stack gap="1.5" minH="0">
      <Box ref={tabListRef} flexShrink={0} onKeyDown={keepTabKeys}>
        <SegmentTabs
          activeId={activeId}
          ariaLabel={t('widgets.preview.details')}
          idBase={idBase}
          isCompact
          tabs={tabs}
          onSelect={setChosenId}
        />
      </Box>
      {failedReads.map((failure) => (
        <LoadError
          key={failure.read}
          flexShrink={0}
          isRetrying={failure.isRetrying}
          layout="inline"
          message={t(failure.read === 'metadata' ? 'widgets.preview.metadataFailed' : 'widgets.preview.workflowFailed')}
          retryLabel={t(
            failure.read === 'metadata' ? 'widgets.preview.retryMetadata' : 'widgets.preview.retryWorkflow'
          )}
          onFocusLost={focusSelectedTab}
          onRetry={failure.retry}
        />
      ))}
      <Stack
        aria-labelledby={segmentTabsTabId(idBase, activeId)}
        gap="0"
        id={segmentTabsPanelId(idBase)}
        minH="0"
        role="tabpanel"
      >
        {!hasActiveContent ? null : activeId === 'details' ? (
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
  onRecall?: (kind: ImageRecallKind) => void;
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
      onRecall?.(recallKind);
    }
  }, [onRecall, recallKind]);
  const recallVerb = recallKind ? getImageRecallVerb(recallKind, t) : null;

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
