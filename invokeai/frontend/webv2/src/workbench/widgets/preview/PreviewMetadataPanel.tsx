import type { GalleryImage, GalleryImageMetadata, GalleryItem, GalleryItemKey } from '@features/gallery';

import { Box, DataList, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { galleryImages, galleryVideos } from '@features/gallery';
import { toGalleryItemKey } from '@features/gallery/contracts';
import { galleryImageMetadataOptions, galleryVideoMetadataOptions } from '@features/gallery/queries';
import { canCopyText, copyText } from '@platform/browser/clipboard';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { IconButton, Scrollable, Tooltip } from '@platform/ui';
import { Button } from '@platform/ui/Button';
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

import {
  parsePreviewMetadata,
  type PreviewMetadataEntry,
  type PreviewMetadataGroup,
  type PreviewMetadataRow,
} from './previewMetadata';

/** Children are keyed by account and identity so closure or identity changes abort supported transports. */

const GROUP_HOVER_VISIBLE = { opacity: 1 };
const HOVER_REVEAL = {
  opacity: 0,
  transitionDuration: 'var(--wb-motion-duration-fast)',
  transitionProperty: 'opacity',
} as const;
// Each reveal answers to its own row: a shared `.group` would show every line's copy on hovering the group row.
const GROUP_ROW_CSS = {
  '& [data-copy-all]': HOVER_REVEAL,
  '&:hover [data-copy-all], & [data-copy-all]:focus-within': GROUP_HOVER_VISIBLE,
} as const;
const GROUP_LINE_CSS = {
  '& [data-line-copy]': HOVER_REVEAL,
  '&:hover [data-line-copy], & [data-line-copy]:focus-within': GROUP_HOVER_VISIBLE,
} as const;
const METADATA_NAMESPACES = ['metadata', 'translation'] as const;

const TAB_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'End', 'Home']);

/** The tab list moves between its tabs with these keys; Preview's arrow bindings must not step the item too. */
const keepTabKeys = (event: KeyboardEvent<HTMLElement>) => {
  if (TAB_KEYS.has(event.key)) {
    event.stopPropagation();
  }
};

/** Field recall only where a dedicated verb exists; others copy and use All/Remix. */
const ENTRY_RECALL_KINDS: Partial<
  Record<string, { capability: keyof ImageRecallCapabilities; kind: ImageRecallKind }>
> = {
  clipSkip: { capability: 'clipSkip', kind: 'clipSkip' },
  // The shared capability gates the Generate target; parsed rows prove the individual prompt exists.
  negativePrompt: { capability: 'prompts', kind: 'negativePrompt' },
  positivePrompt: { capability: 'prompts', kind: 'positivePrompt' },
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
  // Held above the item-keyed query so a group the user expanded stays open while stepping through items.
  const [expandedGroups, setExpandedGroups] = useState<ReadonlySet<string>>(() => new Set());
  const groups = useMemo<GroupExpansion>(
    () => ({
      expanded: expandedGroups,
      toggle: (key) =>
        setExpandedGroups((current) => {
          const next = new Set(current);
          if (!next.delete(key)) {
            next.add(key);
          }
          return next;
        }),
    }),
    [expandedGroups]
  );

  return (
    <PreviewDetailsQuery
      key={`${accountEpoch}:${itemKey}`}
      accountEpoch={accountEpoch}
      actions={actions}
      groups={groups}
      image={image}
      item={item}
      itemKey={itemKey}
    />
  );
};

interface GroupExpansion {
  expanded: ReadonlySet<string>;
  toggle: (key: string) => void;
}

const PreviewDetailsQuery = ({
  accountEpoch,
  actions,
  groups,
  image,
  item,
  itemKey,
}: {
  accountEpoch: number;
  actions: ImageActions;
  groups: GroupExpansion;
  image: GalleryImage | null;
  item: GalleryItem;
  itemKey: GalleryItemKey;
}) =>
  item.kind === 'video' ? (
    <VideoDetailsQuery accountEpoch={accountEpoch} groups={groups} itemKey={itemKey} videoName={item.name} />
  ) : image ? (
    <ImageDetailsQuery accountEpoch={accountEpoch} actions={actions} groups={groups} image={image} itemKey={itemKey} />
  ) : (
    <DetailsTabs graph={null} metadata={null} workflow={null}>
      <ImageDetails
        actions={actions}
        capabilities={EMPTY_IMAGE_RECALL_CAPABILITIES}
        groups={groups}
        image={null}
        isLoading={false}
        metadata={null}
      />
    </DetailsTabs>
  );

/** Metadata from the gallery's shared entry; workflow and graph are Details' own read, failing separately. */
const VideoDetailsQuery = ({
  accountEpoch,
  groups,
  itemKey,
  videoName,
}: {
  accountEpoch: number;
  groups: GroupExpansion;
  itemKey: GalleryItemKey;
  videoName: string;
}) => {
  const { t } = useTranslation(METADATA_NAMESPACES, { nsMode: 'fallback', useSuspense: false });
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
      {entries.length > 0 ? <VideoDetails entries={entries} groups={groups} /> : null}
    </DetailsTabs>
  );
};

const VideoDetails = ({ entries, groups }: { entries: readonly PreviewMetadataRow[]; groups: GroupExpansion }) => (
  <Scrollable flex="1" minH="0">
    <Stack gap="2" pe="1">
      <DataList.Root gap="1.5" orientation="horizontal">
        {entries.map((entry) =>
          'items' in entry ? (
            <MetadataGroupRow key={entry.key} group={entry} groups={groups} />
          ) : (
            <MetadataRow key={entry.key} entry={entry} />
          )
        )}
      </DataList.Root>
    </Stack>
  </Scrollable>
);

/** Metadata from the gallery's shared entry; workflow and graph are Details' own read, failing separately. */
const ImageDetailsQuery = ({
  accountEpoch,
  actions,
  groups,
  image,
  itemKey,
}: {
  accountEpoch: number;
  actions: ImageActions;
  groups: GroupExpansion;
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
        groups={groups}
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
  groups,
  image,
  isLoading,
  metadata,
}: {
  actions: ImageActions;
  capabilities: ImageRecallCapabilities;
  groups: GroupExpansion;
  image: GalleryImage | null;
  isLoading: boolean;
  metadata: GalleryImageMetadata | null;
}) => {
  const { t } = useTranslation(METADATA_NAMESPACES, { nsMode: 'fallback', useSuspense: false });
  const entries: PreviewMetadataRow[] = [
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
              if ('items' in entry) {
                return <MetadataGroupRow key={entry.key} group={entry} groups={groups} />;
              }
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
  const copy = useCopyMetadata();
  const copyValue = useCallback(() => copy(entry.value), [copy, entry.value]);
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

const useCopyMetadata = () => {
  const { t } = useTranslation();

  return useCallback(
    (text: string) =>
      void copyText(text).catch(() =>
        toaster.create({ description: t('common.clipboardBlocked'), title: t('common.couldNotCopy'), type: 'error' })
      ),
    [t]
  );
};

/** One row per array kind: the label once, a summary, the visible lines, and a toggle when some are hidden. */
const MetadataGroupRow = ({ group, groups }: { group: PreviewMetadataGroup; groups: GroupExpansion }) => {
  const { t } = useTranslation();
  const listId = useId();
  const canCopy = canCopyText();
  const copy = useCopyMetadata();
  const isExpanded = groups.expanded.has(group.key);
  const visibleItems = isExpanded || !group.collapsedItems ? group.items : group.collapsedItems;
  const copyAll = useCallback(
    () => copy([...(group.summary ? [group.summary] : []), ...group.items.map((item) => item.value)].join('\n')),
    [copy, group.items, group.summary]
  );
  const { toggle } = groups;
  const toggleExpanded = useCallback(() => toggle(group.key), [group.key, toggle]);
  const copyUnavailable = canCopy ? undefined : t('common.copyUnavailable');

  return (
    <DataList.Item alignItems="start" css={GROUP_ROW_CSS}>
      <DataList.ItemLabel fontSize="xs" minH="6" position="relative">
        <Text as="span">{group.label}</Text>
        {group.summary === undefined && group.count > 1 ? (
          <Text as="span" color="fg.muted" fontVariantNumeric="tabular-nums">
            · {group.count}
          </Text>
        ) : null}
        {/* In the column gap, out of flow: reserving its width would push this row's value past its neighbours'. */}
        <Box data-copy-all="" insetEnd="-4" position="absolute" top="0">
          <IconButton
            tooltip={t('widgets.preview.metadataCopyAll')}
            unavailableReason={copyUnavailable}
            aria-label={t('widgets.preview.metadataCopyAll')}
            color="fg.muted"
            size="sm"
            variant="ghost"
            onClick={copyAll}
          >
            <Icon as={CopyIcon} boxSize="3" />
          </IconButton>
        </Box>
      </DataList.ItemLabel>
      <DataList.ItemValue fontSize="xs" minW="0">
        <Stack align="stretch" flex="1" gap="0.5" minW="0">
          {group.summary ? (
            <Text fontSize="xs" pt="1">
              {group.summary}
            </Text>
          ) : null}
          <Stack as="ul" gap="0" id={listId} listStyleType="none" m="0" p="0">
            {visibleItems.map((item) => (
              <MetadataGroupLine key={item.key} copyUnavailable={copyUnavailable} value={item.value} onCopy={copy} />
            ))}
          </Stack>
          {isExpanded && group.truncationNote ? (
            <Text color="fg.muted" fontSize="xs">
              {group.truncationNote}
            </Text>
          ) : null}
          {group.collapsedItems ? (
            <Button
              alignSelf="flex-start"
              aria-controls={listId}
              aria-expanded={isExpanded}
              color="accent.fg"
              fontSize="xs"
              h="auto"
              px="0"
              size="xs"
              variant="plain"
              onClick={toggleExpanded}
            >
              {isExpanded
                ? t('widgets.preview.metadataShowLess')
                : t('widgets.preview.metadataShowAll', { count: group.count })}
            </Button>
          ) : null}
        </Stack>
      </DataList.ItemValue>
    </DataList.Item>
  );
};

const MetadataGroupLine = ({
  copyUnavailable,
  onCopy,
  value,
}: {
  copyUnavailable: string | undefined;
  onCopy: (text: string) => void;
  value: string;
}) => {
  const { t } = useTranslation();
  const copyValue = useCallback(() => onCopy(value), [onCopy, value]);

  return (
    <HStack as="li" align="flex-start" css={GROUP_LINE_CSS} gap="0" minW="0">
      <Text flex="1" fontSize="xs" minW="0" py="1" whiteSpace="pre-wrap">
        {value}
      </Text>
      <Box data-line-copy="" flexShrink={0}>
        <IconButton
          tooltip={t('common.copy')}
          unavailableReason={copyUnavailable}
          aria-label={t('common.copy')}
          color="fg.muted"
          size="sm"
          variant="ghost"
          onClick={copyValue}
        >
          <Icon as={CopyIcon} boxSize="3" />
        </IconButton>
      </Box>
    </HStack>
  );
};
