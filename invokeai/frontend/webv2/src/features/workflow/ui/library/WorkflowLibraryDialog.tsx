import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type {
  WorkflowLibraryBrowseSnapshot,
  WorkflowLibraryEntry,
  WorkflowLibrarySort,
} from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowLibraryListItem } from '@features/workflow/queries';
import type { WorkflowThumbnailFit } from '@features/workflow/ui/WorkflowUiContext';
import type { ChangeEvent } from 'react';

import {
  Box,
  createListCollection,
  HStack,
  Input,
  Portal,
  Spinner,
  Stack,
  type SystemStyleObject,
  Text,
} from '@chakra-ui/react';
import {
  ensureWorkflowLibraryBrowseLoaded,
  getWorkflowLibraryBrowseSnapshot,
  setWorkflowLibraryBrowseFilter,
  useWorkflowLibraryBrowseSelector,
} from '@features/workflow/data/libraryBrowseStore';
import { useInvocationTemplatesSnapshot } from '@features/workflow/react';
import {
  useWorkflowPreferencesSelector,
  useWorkflowProjectSelector,
  useWorkflowUi,
} from '@features/workflow/ui/WorkflowUiContext';
import {
  requestLibraryCopyChoice,
  setWorkflowLibrarySelection,
  setWorkflowLibraryTab,
  workflowUiStore,
  type WorkflowLibraryTab,
} from '@features/workflow/ui/workflowUiStore';
import { useMountEffect } from '@platform/react/useMountEffect';
import { CloseButton, SegmentTabs, segmentTabsPanelId, segmentTabsTabId } from '@platform/ui';
import { Dialog } from '@platform/ui/Dialog';
import { SegmentedControl } from '@platform/ui/SegmentedControl';
import { Select } from '@platform/ui/Select';
import { Suspense, useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowCardMenuAnchor } from './WorkflowLibraryCard';

import {
  buildLibraryGraphPreviewSource,
  DeferredGraphPreviewDialog,
  preloadGraphPreview,
} from './libraryPreviewSource';
import { ProjectWorkflowsView } from './ProjectWorkflowsView';
import { planLibraryWorkflowOpen, useOpenLibraryWorkflow } from './useOpenLibraryWorkflow';
import { WorkflowLibraryDetailPanel } from './WorkflowLibraryDetailPanel';
import { WorkflowLibraryGrid } from './WorkflowLibraryGrid';
import { WorkflowLibraryTagChips } from './WorkflowLibraryTagChips';
import { useWorkflowLibraryMissingCounts } from './WorkflowRequirementsList';
import { WorkflowThumbnailFitContext } from './workflowThumbnail';

const SEARCH_DEBOUNCE_MS = 300;

const TAB_ITEMS: ReadonlyArray<{ labelKey: string; value: WorkflowLibraryTab }> = [
  { labelKey: 'workflowLibrary.thisProject', value: 'project' },
  { labelKey: 'workflowLibrary.browse', value: 'default' },
  { labelKey: 'workflowLibrary.yours', value: 'user' },
];

const SORT_ITEMS: ReadonlyArray<{ labelKey: string; value: WorkflowLibrarySort }> = [
  { labelKey: 'workflowLibrary.sortUpdated', value: 'updated' },
  { labelKey: 'workflowLibrary.sortOpened', value: 'opened' },
  { labelKey: 'workflowLibrary.sortCreated', value: 'created' },
  { labelKey: 'workflowLibrary.sortName', value: 'name' },
];

const THUMBNAIL_FIT_ITEMS: ReadonlyArray<{ labelKey: string; value: WorkflowThumbnailFit }> = [
  { labelKey: 'workflowLibrary.thumbnailFitCover', value: 'cover' },
  { labelKey: 'workflowLibrary.thumbnailFitContain', value: 'contain' },
];

const SORT_SELECT_WIDTH = '9.5rem';

/**
 * Title, search, tabs and Close need about 36rem side by side. Below that header width Close stays beside the title,
 * the search and the tabs take full rows of their own (in focus order), and the tag chips sit above Sort and Fit.
 */
const NARROW_HEADER = '@container workflow-library-header (width < 36rem)';
const HEADER_CSS: SystemStyleObject = { containerName: 'workflow-library-header', containerType: 'inline-size' };
const HEADER_TOP_CSS: SystemStyleObject = {
  alignItems: 'center',
  columnGap: '3',
  display: 'grid',
  gridTemplateAreas: '"title lead tabs close"',
  gridTemplateColumns: 'auto minmax(0, 1fr) auto auto',
  rowGap: '2',
  [NARROW_HEADER]: {
    gridTemplateAreas: '"title close" "lead lead" "tabs tabs"',
    gridTemplateColumns: 'minmax(0, 1fr) auto',
  },
};
const HEADER_FILTERS_CSS: SystemStyleObject = {
  [NARROW_HEADER]: { flexWrap: 'wrap', justifyContent: 'flex-end', rowGap: '2' },
};
// An empty strip (the project tab, or no tags) keeps only its spacer role, so it adds no row.
const TAG_STRIP_CSS: SystemStyleObject = { [NARROW_HEADER]: { flexBasis: 'full', '&:empty': { flexBasis: '0' } } };
const SORT_SELECT_CSS: SystemStyleObject = { [NARROW_HEADER]: { flex: '1', minW: '0', w: 'auto' } };

/** Flat and shallow-comparable, so unrelated store patches do not re-render the shell. */
const selectBrowseView = (snapshot: WorkflowLibraryBrowseSnapshot) => ({
  category: snapshot.filter.category,
  entries: snapshot.entries,
  error: snapshot.error,
  status: snapshot.status,
  tag: snapshot.filter.tag,
  tagCounts: snapshot.tagCounts,
});

const selectLibraryNavigation = (snapshot: ReturnType<typeof workflowUiStore.getSnapshot>) => ({
  selection: snapshot.librarySelection,
  tab: snapshot.libraryTab,
});

/**
 * On a template tab's first open, load the first page and choose defaults for empty accounts only if filters have
 * not changed during the probe. The project view never loads library pages.
 */
const WorkflowLibraryBrowseSession = ({ sort }: { sort: WorkflowLibrarySort }) => {
  useMountEffect(() => {
    void ensureWorkflowLibraryBrowseLoaded(sort).then(() => {
      const { filter, userTotal } = getWorkflowLibraryBrowseSnapshot();

      if (userTotal === 0 && filter.category === 'user' && !filter.search) {
        setWorkflowLibraryBrowseFilter({ category: 'default', tag: null });
        setWorkflowLibraryTab('default');
      }
    });
  });

  return null;
};

/** A preview request from either view: a library entry's enriched document or a project workflow's own. */
type PreviewRequest =
  | { kind: 'library'; entry: WorkflowLibraryEntry }
  | { kind: 'project'; entry: ProjectWorkflowEntry };

export const WorkflowLibraryDialog = ({
  isOpen,
  onOpenChange,
}: {
  isOpen: boolean;
  onOpenChange: (isOpen: boolean) => void;
}) => {
  const { t } = useTranslation();
  const { category, entries, error, status, tag, tagCounts } = useWorkflowLibraryBrowseSelector(selectBrowseView);
  const { selection, tab } = workflowUiStore.useSelector(selectLibraryNavigation);
  const projectId = useWorkflowProjectSelector((project) => project.id);
  const projectWorkflows = useWorkflowProjectSelector((project) => project.workflows);
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const { setLibraryPreferences } = useWorkflowUi();
  const sort = useWorkflowPreferencesSelector((preferences) => preferences.workflowLibrarySort);
  const thumbnailFit = useWorkflowPreferencesSelector((preferences) => preferences.workflowLibraryThumbnailFit);
  const [searchInput, setSearchInput] = useState('');
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string | null>(null);
  // Clear pending preview entries on every library-close path because the persistent shell otherwise resurrects
  // them on reopen.
  const [previewRequest, setPreviewRequest] = useState<PreviewRequest | null>(null);
  // Close before clearing the request so the lazy dialog remains mounted through its exit transition.
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [contextMenuPoint, setContextMenuPoint] = useState<WorkflowCardMenuAnchor | null>(null);
  // The element focus returns to when the menu closes. It outlives the open menu: the menu is told it closed only
  // after React has rendered the close, so the trigger it restores focus to must still be the one that opened it.
  const [contextMenuTriggerId, setContextMenuTriggerId] = useState<string | null>(null);
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const closeDialog = useCallback(() => {
    // Closing the library removes its preview immediately; the parent surface is leaving too.
    setPreviewRequest(null);
    setIsPreviewOpen(false);
    setContextMenuPoint(null);
    onOpenChange(false);
  }, [onOpenChange]);
  const { loadPhase, open } = useOpenLibraryWorkflow(closeDialog);
  const isLoadPending = loadPhase !== 'idle';
  const missingCounts = useWorkflowLibraryMissingCounts(entries);

  // Derive a fallback selection when filtering/deletion removes the selected row.
  const activeWorkflowId = entries.some((entry) => entry.item.workflow_id === selectedWorkflowId)
    ? selectedWorkflowId
    : (entries[0]?.item.workflow_id ?? null);
  const activeEntry = entries.find((entry) => entry.item.workflow_id === activeWorkflowId) ?? null;
  // The project selection belongs to the project it was made in; another project starts from its active workflow.
  const projectSelectionId = selection?.projectId === projectId ? selection.workflowId : null;

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
    setPreviewRequest({ entry, kind: 'library' });
    setIsPreviewOpen(true);
  }, []);
  const handleProjectPreviewRequest = useCallback((entry: ProjectWorkflowEntry) => {
    setPreviewRequest({ entry, kind: 'project' });
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
      setPreviewRequest(null);
    }
  }, [isPreviewOpen]);

  // Only ready enrichment has a document; guard stale preview entries after revalidation.
  const previewSource = useMemo(() => {
    if (!previewRequest || templatesSnapshot.status !== 'loaded') {
      return null;
    }

    if (previewRequest.kind === 'project') {
      return buildLibraryGraphPreviewSource(previewRequest.entry.document, templatesSnapshot.templates);
    }

    return previewRequest.entry.enrichment.status === 'ready'
      ? buildLibraryGraphPreviewSource(previewRequest.entry.enrichment.document, templatesSnapshot.templates)
      : null;
  }, [previewRequest, templatesSnapshot]);
  const previewGraphId =
    previewRequest?.kind === 'project'
      ? previewRequest.entry.document.id
      : (previewRequest?.entry.item.workflow_id ?? '');
  const previewLabel =
    (previewRequest?.kind === 'project' ? previewRequest.entry.document.name : previewRequest?.entry.item.name) ||
    t('workflowLibrary.untitled');

  const handleSearchChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const { value } = event.currentTarget;

    setSearchInput(value);

    if (searchTimerRef.current !== null) {
      clearTimeout(searchTimerRef.current);
    }

    // A post-unmount search debounce may update the browse store for the next opening.
    searchTimerRef.current = setTimeout(() => setWorkflowLibraryBrowseFilter({ search: value }), SEARCH_DEBOUNCE_MS);
  }, []);

  const handleTabChange = useCallback(
    (value: WorkflowLibraryTab) => {
      setContextMenuPoint(null);

      if (value === 'project') {
        setWorkflowLibraryTab('project');
        return;
      }

      // The store applies filter patches literally, so clearing the tag when the
      // category changes (its chips do not carry over) is the UI's job.
      setWorkflowLibraryTab(value);

      if (category !== value) {
        setWorkflowLibraryBrowseFilter({ category: value, tag: null });
      }
    },
    [category]
  );

  const handleTagSelect = useCallback((nextTag: string | null) => setWorkflowLibraryBrowseFilter({ tag: nextTag }), []);
  const sortCollection = useMemo(
    () => createListCollection({ items: SORT_ITEMS.map((item) => ({ label: t(item.labelKey), value: item.value })) }),
    [t]
  );
  const sortValue = useMemo(() => [sort], [sort]);
  const handleSortChange = useCallback(
    ({ value }: { value: string[] }) => {
      const next = SORT_ITEMS.find((item) => item.value === value[0])?.value;

      if (next) {
        setLibraryPreferences({ workflowLibrarySort: next });
        setWorkflowLibraryBrowseFilter({ sort: next });
      }
    },
    [setLibraryPreferences]
  );
  const thumbnailFitOptions = useMemo(
    () => THUMBNAIL_FIT_ITEMS.map((item) => ({ label: t(item.labelKey), value: item.value })),
    [t]
  );
  const handleThumbnailFitChange = useCallback(
    (value: string) =>
      setLibraryPreferences({ workflowLibraryThumbnailFit: value === 'contain' ? 'contain' : 'cover' }),
    [setLibraryPreferences]
  );

  // The first copy is added directly; with one already in the project, the host asks what opening should do.
  const handleOpenItem = useCallback(
    (item: WorkflowLibraryListItem) => {
      if (planLibraryWorkflowOpen(projectWorkflows, item.workflow_id).kind === 'choose') {
        requestLibraryCopyChoice(projectId, item);
        return;
      }

      void open(item, 'first-copy');
    },
    [open, projectId, projectWorkflows]
  );
  const handleOpenWorkflow = useCallback(
    (workflowId: string) => {
      const entry = getWorkflowLibraryBrowseSnapshot().entries.find(
        (candidate) => candidate.item.workflow_id === workflowId
      );

      if (entry) {
        handleOpenItem(entry.item);
      }
    },
    [handleOpenItem]
  );

  const handleDeleted = useCallback(() => {
    setSelectedWorkflowId(null);
    setContextMenuPoint(null);
  }, []);
  const handleCardContextMenu = useCallback(
    (workflowId: string, point: WorkflowCardMenuAnchor) => {
      if (tab === 'project') {
        setWorkflowLibrarySelection({ projectId, workflowId });
      } else {
        setSelectedWorkflowId(workflowId);
      }

      setContextMenuTriggerId(point.kind === 'trigger' ? point.triggerId : null);

      // The menu button that opened the menu closes it again.
      if (
        point.kind === 'trigger' &&
        contextMenuPoint?.kind === 'trigger' &&
        contextMenuPoint.triggerId === point.triggerId
      ) {
        setContextMenuPoint(null);
        return;
      }

      // An open menu does not follow a new anchor: close it and reopen it at the
      // new point once that close has rendered, as a native menu relocates.
      if (contextMenuPoint) {
        setContextMenuPoint(null);
        requestAnimationFrame(() => setContextMenuPoint(point));
      } else {
        setContextMenuPoint(point);
      }
    },
    [contextMenuPoint, projectId, tab]
  );
  const closeContextMenu = useCallback(() => setContextMenuPoint(null), []);
  const handleProjectSelect = useCallback(
    (workflowId: string | null) => setWorkflowLibrarySelection(workflowId ? { projectId, workflowId } : null),
    [projectId]
  );

  const isProjectTab = tab === 'project';
  const tabsIdBase = useId();
  const activeTab: WorkflowLibraryTab = isProjectTab ? 'project' : category;
  const tabItems = useMemo(() => TAB_ITEMS.map((item) => ({ id: item.value, label: t(item.labelKey) })), [t]);

  return (
    <>
      <Dialog.Root open={isOpen} size="xl" onOpenChange={handleDialogOpenChange}>
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content
              ref={preloadGraphPreview}
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
                  <Spinner color="accent.solid" size="2xl" />
                  <Text fontSize="md" fontWeight="600">
                    {loadPhase === 'fetching' ? t('workflowLibrary.fetching') : t('workflowLibrary.applying')}
                  </Text>
                </Stack>
              ) : null}
              <Dialog.Header>
                <Stack css={HEADER_CSS} gap="2" minW="0" w="full">
                  <Box css={HEADER_TOP_CSS} minW="0">
                    <Dialog.Title gridArea="title" minW="0" truncate>
                      {t('workflowLibrary.title')}
                    </Dialog.Title>
                    {isProjectTab ? (
                      <Text color="fg.subtle" fontSize="md" gridArea="lead" minW="0" truncate>
                        {t('workflowLibrary.thisProjectHint')}
                      </Text>
                    ) : (
                      <Input
                        aria-label={t('workflowLibrary.searchPlaceholder')}
                        gridArea="lead"
                        minW="0"
                        placeholder={t('workflowLibrary.searchPlaceholder')}
                        type="search"
                        value={searchInput}
                        onChange={handleSearchChange}
                      />
                    )}
                    <Box gridArea="tabs" minW="0">
                      <SegmentTabs
                        activeId={activeTab}
                        ariaLabel={t('workflowLibrary.title')}
                        idBase={tabsIdBase}
                        isCompact
                        tabs={tabItems}
                        onSelect={handleTabChange}
                      />
                    </Box>
                    <Dialog.CloseTrigger asChild>
                      <CloseButton
                        disabled={isLoadPending}
                        gridArea="close"
                        insetEnd="auto"
                        position="static"
                        top="auto"
                      />
                    </Dialog.CloseTrigger>
                  </Box>
                  <HStack css={HEADER_FILTERS_CSS} gap="3" minW="0">
                    <Box css={TAG_STRIP_CSS} flex="1" minW="0">
                      {isProjectTab ? null : (
                        <WorkflowLibraryTagChips selectedTag={tag} tagCounts={tagCounts} onSelect={handleTagSelect} />
                      )}
                    </Box>
                    {/* Bundled templates always list by name, and project workflows keep their own order. */}
                    {category === 'user' && !isProjectTab ? (
                      <Select
                        aria-label={t('workflowLibrary.sortLabel')}
                        collection={sortCollection}
                        css={SORT_SELECT_CSS}
                        flexShrink={0}
                        size="sm"
                        value={sortValue}
                        w={SORT_SELECT_WIDTH}
                        onValueChange={handleSortChange}
                      />
                    ) : null}
                    <SegmentedControl
                      ariaLabel={t('workflowLibrary.thumbnailFit')}
                      flexShrink={0}
                      isFullWidth={false}
                      options={thumbnailFitOptions}
                      size="sm"
                      value={thumbnailFit}
                      onChange={handleThumbnailFitChange}
                    />
                  </HStack>
                </Stack>
              </Dialog.Header>
              <Dialog.Body
                aria-labelledby={segmentTabsTabId(tabsIdBase, activeTab)}
                data-library-tab={tab}
                data-pending-preview={previewGraphId || undefined}
                display="flex"
                flex="1"
                gap="3"
                id={segmentTabsPanelId(tabsIdBase)}
                minH="0"
                role="tabpanel"
              >
                <WorkflowThumbnailFitContext value={thumbnailFit}>
                  {isProjectTab ? (
                    <ProjectWorkflowsView
                      contextMenuPoint={contextMenuPoint}
                      contextMenuTriggerId={contextMenuTriggerId}
                      selectedWorkflowId={projectSelectionId}
                      onClose={closeDialog}
                      onContextMenu={handleCardContextMenu}
                      onContextMenuClose={closeContextMenu}
                      onPreview={handleProjectPreviewRequest}
                      onSelect={handleProjectSelect}
                    />
                  ) : (
                    <>
                      <WorkflowLibraryGrid
                        entries={entries}
                        error={error}
                        missingCounts={missingCounts}
                        openMenuAnchor={contextMenuPoint}
                        selectedWorkflowId={activeWorkflowId}
                        status={status}
                        onContextMenu={handleCardContextMenu}
                        onOpen={handleOpenWorkflow}
                        onSelect={setSelectedWorkflowId}
                      />
                      <WorkflowLibraryDetailPanel
                        contextMenuPoint={contextMenuPoint}
                        contextMenuTriggerId={contextMenuTriggerId}
                        entry={activeEntry}
                        projectWorkflows={projectWorkflows}
                        onClose={closeDialog}
                        onContextMenuClose={closeContextMenu}
                        onDeleted={handleDeleted}
                        onDuplicated={setSelectedWorkflowId}
                        onOpen={handleOpenItem}
                        onPreview={handlePreviewRequest}
                      />
                    </>
                  )}
                </WorkflowThumbnailFitContext>
              </Dialog.Body>
              {isOpen && !isProjectTab ? <WorkflowLibraryBrowseSession sort={sort} /> : null}
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
      {previewRequest && previewSource ? (
        <Suspense fallback={null}>
          <DeferredGraphPreviewDialog
            graphId={previewGraphId}
            hideInvoke
            isOpen={isPreviewOpen}
            source={previewSource}
            sourceLabel={previewLabel}
            onExitComplete={handlePreviewExitComplete}
            onOpenChange={handlePreviewOpenChange}
          />
        </Suspense>
      ) : null}
    </>
  );
};
