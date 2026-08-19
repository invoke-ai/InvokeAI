import type { WorkflowLibraryBrowseSnapshot, WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowLibraryListItem } from '@features/workflow/queries';
import type { ChangeEvent } from 'react';

import { Dialog, HStack, Input, Portal, SegmentGroup, Spinner, Stack, Text } from '@chakra-ui/react';
import {
  ensureWorkflowLibraryBrowseLoaded,
  getWorkflowLibraryBrowseSnapshot,
  setWorkflowLibraryBrowseFilter,
  useWorkflowLibraryBrowseSelector,
} from '@features/workflow/data/libraryBrowseStore';
import { useInvocationTemplatesSnapshot } from '@features/workflow/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { CloseButton } from '@platform/ui';
import { lazy, Suspense, useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { buildLibraryGraphPreviewSource } from './libraryPreviewSource';
import { useLoadLibraryWorkflow } from './useLoadLibraryWorkflow';
import { WorkflowLibraryDetailPanel } from './WorkflowLibraryDetailPanel';
import { WorkflowLibraryGrid } from './WorkflowLibraryGrid';
import { WorkflowLibraryTagChips } from './WorkflowLibraryTagChips';
import { useWorkflowLibraryMissingCounts } from './WorkflowRequirementsList';

/**
 * xyflow (~174 KB) stays out of this dialog's own chunk: the preview dialog
 * is only ever needed once a card asks to preview it, so it is dynamic
 * `import()`ed here rather than statically imported like every other panel
 * in this file (mirrors `WidgetActionsMenu.tsx`'s `GraphPreviewHost`).
 */
const LazyGraphPreviewDialog = lazy(() =>
  import('@features/workflow/ui/graph-preview/GraphPreviewDialog').then((module) => ({
    default: module.GraphPreviewDialog,
  }))
);

/**
 * Backend workflow library browser. Filtering and paging are server side and
 * live in the browse store, so this shell owns exactly two pieces of state:
 * the raw text in the search box (debounced into the store) and which card is
 * selected. Everything else is read back out of the store.
 */

const SEARCH_DEBOUNCE_MS = 300;

const CATEGORY_ITEMS = [
  { labelKey: 'workflowLibrary.browse', value: 'default' },
  { labelKey: 'workflowLibrary.yours', value: 'user' },
] as const;

/** Flat and shallow-comparable, so unrelated store patches do not re-render the shell. */
const selectBrowseView = (snapshot: WorkflowLibraryBrowseSnapshot) => ({
  category: snapshot.filter.category,
  entries: snapshot.entries,
  error: snapshot.error,
  status: snapshot.status,
  tag: snapshot.filter.tag,
  tagCounts: snapshot.tagCounts,
});

/**
 * Mounted only while the dialog is open, so its mount effect *is* the "dialog
 * opened" hook: it kicks the first page load and, on a fresh install with no
 * saved workflows, lands the user on the bundled defaults instead of an empty
 * "Yours". The switch is skipped if the user has already moved the filter
 * while the probe was in flight.
 */
const WorkflowLibraryBrowseSession = () => {
  useMountEffect(() => {
    void ensureWorkflowLibraryBrowseLoaded().then(() => {
      const { filter, userTotal } = getWorkflowLibraryBrowseSnapshot();

      if (userTotal === 0 && filter.category === 'user' && !filter.search) {
        setWorkflowLibraryBrowseFilter({ category: 'default', tag: null });
      }
    });
  });

  return null;
};

export const WorkflowLibraryDialog = ({
  isOpen,
  onOpenChange,
}: {
  isOpen: boolean;
  onOpenChange: (isOpen: boolean) => void;
}) => {
  const { t } = useTranslation();
  const { category, entries, error, status, tag, tagCounts } = useWorkflowLibraryBrowseSelector(selectBrowseView);
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const [searchInput, setSearchInput] = useState('');
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string | null>(null);
  // The record of which workflow the rail asked to preview — mounts the lazy
  // preview dialog below while set. This dialog shell is never unmounted while
  // the app is up (only its `isOpen` toggles), so every path that can close
  // *this* dialog has to clear it too, or a stale preview would resurrect
  // itself the next time the library opens.
  const [previewEntry, setPreviewEntry] = useState<WorkflowLibraryEntry | null>(null);
  // Tracked separately from `previewEntry` so closing the preview can play its
  // exit transition: `isPreviewOpen` goes false first, and the entry (which is
  // what keeps the lazy dialog mounted) is only released once the transition
  // has finished.
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const closeDialog = useCallback(() => {
    // The library itself is leaving, and it takes the preview with it — there
    // is nothing left to animate against, so this drops the mount outright.
    setPreviewEntry(null);
    setIsPreviewOpen(false);
    onOpenChange(false);
  }, [onOpenChange]);
  const { load, loadPhase } = useLoadLibraryWorkflow(closeDialog);
  const isLoadPending = loadPhase !== 'idle';
  const missingCounts = useWorkflowLibraryMissingCounts(entries);

  // Selection is derived, not stored: a filter change or a deletion can retire
  // the selected row, and the head of the list takes over without an effect.
  const activeWorkflowId = entries.some((entry) => entry.item.workflow_id === selectedWorkflowId)
    ? selectedWorkflowId
    : (entries[0]?.item.workflow_id ?? null);
  const activeEntry = entries.find((entry) => entry.item.workflow_id === activeWorkflowId) ?? null;

  const handleDialogOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (isLoadPending) {
        return;
      }

      if (event.open) {
        onOpenChange(true);
      } else {
        closeDialog();
      }
    },
    [isLoadPending, onOpenChange, closeDialog]
  );

  const handlePreviewRequest = useCallback((entry: WorkflowLibraryEntry) => {
    setPreviewEntry(entry);
    setIsPreviewOpen(true);
  }, []);

  const handlePreviewOpenChange = useCallback((open: boolean) => {
    if (!open) {
      setIsPreviewOpen(false);
    }
  }, []);

  // Guarded on `isPreviewOpen`: previewing another card while the last one is
  // still animating out re-opens the same dialog, and an exit report that
  // arrives after that must not pull the mount out from under it.
  const handlePreviewExitComplete = useCallback(() => {
    if (!isPreviewOpen) {
      setPreviewEntry(null);
    }
  }, [isPreviewOpen]);

  // Only a `'ready'` enrichment carries the compiled document; the rail's
  // Preview action is disabled for anything else, so this is a defensive
  // fallback (a stale `previewEntry` from before a revalidation), not a path
  // the UI can normally reach.
  const previewSource = useMemo(() => {
    if (!previewEntry || previewEntry.enrichment.status !== 'ready' || templatesSnapshot.status !== 'loaded') {
      return null;
    }

    return buildLibraryGraphPreviewSource(previewEntry.enrichment.document, templatesSnapshot.templates);
  }, [previewEntry, templatesSnapshot]);

  const handleSearchChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const { value } = event.currentTarget;

    setSearchInput(value);

    if (searchTimerRef.current !== null) {
      clearTimeout(searchTimerRef.current);
    }

    // Deliberately not a `useEffect` cleanup: a debounce that fires after
    // unmount only patches the store the next open would reload anyway.
    searchTimerRef.current = setTimeout(() => setWorkflowLibraryBrowseFilter({ search: value }), SEARCH_DEBOUNCE_MS);
  }, []);

  const handleCategoryChange = useCallback((event: { value: string | null }) => {
    // The store applies filter patches literally, so clearing the tag when the
    // category changes (its chips do not carry over) is the UI's job.
    if (event.value === 'default' || event.value === 'user') {
      setWorkflowLibraryBrowseFilter({ category: event.value, tag: null });
    }
  }, []);

  const handleTagSelect = useCallback((nextTag: string | null) => setWorkflowLibraryBrowseFilter({ tag: nextTag }), []);

  const handleOpenWorkflow = useCallback(
    (workflowId: string) => {
      const entry = getWorkflowLibraryBrowseSnapshot().entries.find(
        (candidate) => candidate.item.workflow_id === workflowId
      );

      if (entry) {
        void load(entry.item);
      }
    },
    [load]
  );

  const handleOpenItem = useCallback((item: WorkflowLibraryListItem) => void load(item), [load]);

  // The deleted row is gone from the next refresh; dropping the selection lets
  // the head of the list take over, the same way a filter change does.
  const handleDeleted = useCallback(() => setSelectedWorkflowId(null), []);

  return (
    <>
      <Dialog.Root open={isOpen} placement="center" size="xl" onOpenChange={handleDialogOpenChange}>
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content
              aria-busy={isLoadPending}
              h="80vh"
              maxH="80vh"
              maxW="min(72rem, calc(100vw - 4rem))"
              position="relative"
            >
              {isLoadPending ? (
                <Stack
                  alignItems="center"
                  aria-live="polite"
                  bg="bg/85"
                  inset="0"
                  justifyContent="center"
                  position="absolute"
                  role="status"
                  zIndex="modal"
                >
                  <Spinner color="accent.solid" size="lg" />
                  <Text fontSize="xs" fontWeight="600">
                    {loadPhase === 'fetching' ? t('workflowLibrary.fetching') : t('workflowLibrary.applying')}
                  </Text>
                </Stack>
              ) : null}
              <Dialog.Header>
                <Stack gap="2" minW="0" w="full">
                  <HStack gap="3" minW="0">
                    <Dialog.Title flexShrink={0}>{t('workflowLibrary.title')}</Dialog.Title>
                    <Input
                      aria-label={t('workflowLibrary.searchPlaceholder')}
                      flex="1"
                      minW="0"
                      placeholder={t('workflowLibrary.searchPlaceholder')}
                      size="xs"
                      type="search"
                      value={searchInput}
                      onChange={handleSearchChange}
                    />
                    <SegmentGroup.Root flexShrink={0} size="xs" value={category} onValueChange={handleCategoryChange}>
                      <SegmentGroup.Indicator />
                      {CATEGORY_ITEMS.map((item) => (
                        <SegmentGroup.Item key={item.value} value={item.value}>
                          <SegmentGroup.ItemHiddenInput />
                          <SegmentGroup.ItemText>{t(item.labelKey)}</SegmentGroup.ItemText>
                        </SegmentGroup.Item>
                      ))}
                    </SegmentGroup.Root>
                    {/* In the header row rather than the dialog's absolutely
                        positioned corner: with a second header row of tag chips
                        underneath, the corner placement floated the control
                        across both bands instead of reading as part of either. */}
                    <Dialog.CloseTrigger asChild>
                      <CloseButton
                        color="fg.muted"
                        disabled={isLoadPending}
                        flexShrink={0}
                        insetEnd="auto"
                        position="static"
                        size="sm"
                        top="auto"
                      />
                    </Dialog.CloseTrigger>
                  </HStack>
                  <WorkflowLibraryTagChips selectedTag={tag} tagCounts={tagCounts} onSelect={handleTagSelect} />
                </Stack>
              </Dialog.Header>
              <Dialog.Body
                data-pending-preview={previewEntry?.item.workflow_id}
                display="flex"
                flex="1"
                gap="3"
                minH="0"
              >
                <WorkflowLibraryGrid
                  entries={entries}
                  error={error}
                  missingCounts={missingCounts}
                  selectedWorkflowId={activeWorkflowId}
                  status={status}
                  onOpen={handleOpenWorkflow}
                  onSelect={setSelectedWorkflowId}
                />
                <WorkflowLibraryDetailPanel
                  entry={activeEntry}
                  onClose={closeDialog}
                  onDeleted={handleDeleted}
                  onDuplicated={setSelectedWorkflowId}
                  onOpen={handleOpenItem}
                  onPreview={handlePreviewRequest}
                />
              </Dialog.Body>
              {isOpen ? <WorkflowLibraryBrowseSession /> : null}
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
      {previewEntry && previewSource ? (
        <Suspense fallback={null}>
          <LazyGraphPreviewDialog
            graphId={previewEntry.item.workflow_id}
            hideInvoke
            isOpen={isPreviewOpen}
            source={previewSource}
            sourceLabel={previewEntry.item.name || t('workflowLibrary.untitled')}
            onExitComplete={handlePreviewExitComplete}
            onOpenChange={handlePreviewOpenChange}
          />
        </Suspense>
      ) : null}
    </>
  );
};
