import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type {
  WorkflowLibraryBrowseFilter,
  WorkflowLibraryBrowseSnapshot,
  WorkflowLibraryEntry,
  WorkflowLibraryPlace,
  WorkflowLibrarySort,
} from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowLibraryListItem } from '@features/workflow/queries';
import type { WorkflowLibraryLayout, WorkflowThumbnailFit } from '@features/workflow/ui/WorkflowUiContext';

import { Box, HStack, Portal, Spinner, Stack, type SystemStyleObject, Text } from '@chakra-ui/react';
import { useOptionalGalleryHostProjectContext } from '@features/gallery/react';
import { useAuthSession } from '@features/identity';
import {
  ensureWorkflowLibraryBrowseLoaded,
  getWorkflowLibraryBrowseSnapshot,
  getWorkflowLibraryOrder,
  getWorkflowLibrarySortDefaultDirection,
  holdWorkflowLibraryBrowse,
  loadWorkflowLibraryPlaceTotals,
  selectWorkflowLibraryEntry,
  setWorkflowLibraryBrowseFilter,
  useWorkflowLibraryBrowseSelector,
} from '@features/workflow/data/libraryBrowseStore';
import { useInvocationTemplatesSnapshot } from '@features/workflow/react';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import {
  useWorkflowPreferencesSelector,
  useWorkflowProjectSelector,
  useWorkflowUi,
} from '@features/workflow/ui/WorkflowUiContext';
import {
  requestLibraryCopyChoice,
  setWorkflowLibraryProjectOrder,
  setWorkflowLibrarySearch,
  setWorkflowLibrarySelection,
  setWorkflowLibraryTab,
  setWorkflowLibraryTags,
  workflowUiStore,
  type ProjectWorkflowSort,
  type WorkflowLibraryTab,
} from '@features/workflow/ui/workflowUiStore';
import { useMountEffect } from '@platform/react/useMountEffect';
import { CloseButton } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { Suspense, useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowCardMenuAnchor } from './WorkflowLibraryCard';
import type { WorkflowLibraryListTrailingField } from './WorkflowLibraryList';
import type { WorkflowLibraryCardSize } from './workflowThumbnail';

import {
  buildLibraryGraphPreviewSource,
  DeferredGraphPreviewDialog,
  preloadGraphPreview,
} from './libraryPreviewSource';
import {
  countProjectWorkflowTags,
  queryProjectWorkflowEntries,
  toProjectWorkflowLibraryEntries,
} from './projectWorkflowEntries';
import { ProjectWorkflowsView } from './ProjectWorkflowsView';
import { planLibraryWorkflowOpen, useOpenLibraryWorkflow } from './useOpenLibraryWorkflow';
import { isWorkflowDetailSheetEscape } from './WorkflowDetailSheet';
import { WorkflowLibraryBrowsePanes, WorkflowLibraryResultLine } from './WorkflowLibraryBrowsePanes';
import {
  getWorkflowLibraryPlaceLabelId,
  WORKFLOW_LIBRARY_NARROW,
  WORKFLOW_LIBRARY_WIDE,
  WorkflowLibraryPlaceSelect,
  WorkflowLibraryRail,
  WorkflowLibraryTagButton,
  WorkflowLibraryTagChips,
} from './WorkflowLibraryRail';
import { WorkflowLibraryToolbar, type WorkflowLibrarySortOption } from './WorkflowLibraryToolbar';
import { useWorkflowLibraryMissingCounts } from './WorkflowRequirementsList';
import { WorkflowLibraryCardSizeContext, WorkflowThumbnailFitContext } from './workflowThumbnail';

const SEARCH_DEBOUNCE_MS = 300;
// Short windows need the panes' full shared height; reclaim modal margins.
const CONTENT_CSS: SystemStyleObject = {
  containerName: 'workflow-library-dialog',
  containerType: 'inline-size',
  '@media (height < 40rem)': {
    h: 'calc(100dvh - 2rem)',
    maxH: 'calc(100dvh - 2rem)',
    maxW: 'min(80rem, calc(100vw - 1rem))',
  },
};
/**
 * Wide (`WORKFLOW_LIBRARY_WIDE`): the body is one grid, matching the artboard. The places rail and the detail rail
 * are flush full-height columns from under the header to the dialog's bottom edge, each separated by a 1px border;
 * the toolbar, result line and cards share the centre column between them. The centre and panes boxes dissolve
 * (`display: contents`) so their parts become the grid's items, and the body is the container the cards and detail
 * rail lay out against (`workflow-library-content`); a body too short for them scrolls as a whole.
 */
const BODY_CSS: SystemStyleObject = {
  display: 'flex',
  [WORKFLOW_LIBRARY_WIDE]: {
    columnGap: '3',
    containerName: 'workflow-library-content',
    containerType: 'size',
    display: 'grid',
    gridTemplateAreas: '"rail toolbar detail" "rail results detail" "rail panes detail"',
    gridTemplateColumns: 'auto minmax(0, 1fr) auto',
    gridTemplateRows: 'auto auto minmax(0, 1fr)',
    overflowY: 'auto',
    rowGap: '2',
    '& [data-workflow-library-toolbar]': { gridArea: 'toolbar', mt: '3' },
    '& [data-workflow-library-results]': { gridArea: 'results' },
  },
};
const RAIL_SLOT_CSS: SystemStyleObject = {
  [WORKFLOW_LIBRARY_NARROW]: { display: 'none' },
  [WORKFLOW_LIBRARY_WIDE]: { gridArea: 'rail' },
};
/**
 * Narrow (`WORKFLOW_LIBRARY_NARROW`): the centre column is the container the cards and the sheet lay out against, and
 * it scrolls when they cannot share its height; the places rail gives way to the toolbar's select and Tags button.
 */
const CENTRE_CSS: SystemStyleObject = {
  containerName: 'workflow-library-content',
  containerType: 'size',
  display: 'flex',
  flex: '1',
  flexDirection: 'column',
  gap: '2',
  minH: '0',
  minW: '0',
  overflowY: 'auto',
  p: '3',
  // When the column scrolls (a flow layout), scrolling into view keeps clear of its padding and of the sheet's sticky
  // summary.
  [WORKFLOW_LIBRARY_NARROW]: { scrollPaddingBottom: '6rem', scrollPaddingTop: '3' },
  [WORKFLOW_LIBRARY_WIDE]: { containerType: 'normal', display: 'contents' },
};

const LIBRARY_SORTS: readonly WorkflowLibrarySort[] = ['opened', 'updated', 'lastRun', 'created', 'name'];
const PROJECT_SORTS: readonly ProjectWorkflowSort[] = ['order', 'name', 'updated', 'lastRun'];
const SORT_LABEL_KEYS: Record<WorkflowLibrarySort | ProjectWorkflowSort, string> = {
  created: 'workflowLibrary.sortCreated',
  lastRun: 'workflowLibrary.sortLastRun',
  name: 'workflowLibrary.sortName',
  opened: 'workflowLibrary.sortOpened',
  order: 'workflowLibrary.sortProjectOrder',
  updated: 'workflowLibrary.sortUpdated',
};

/** What a direction means for a key: dates newest or oldest first, names A to Z, the project's order or reversed. */
const getDirectionLabelKey = (sort: WorkflowLibrarySort | ProjectWorkflowSort, direction: 'asc' | 'desc') => {
  if (sort === 'name') {
    return direction === 'asc' ? 'workflowLibrary.sortAToZ' : 'workflowLibrary.sortZToA';
  }

  if (sort === 'order') {
    return direction === 'asc' ? 'workflowLibrary.sortFirstToLast' : 'workflowLibrary.sortLastToFirst';
  }

  return direction === 'asc' ? 'workflowLibrary.sortOldestFirst' : 'workflowLibrary.sortNewestFirst';
};

const flipDirection = (direction: 'asc' | 'desc') => (direction === 'asc' ? 'desc' : 'asc');

/** The date a list row shows: the one the current order sorts by, else when the workflow last changed. */
const TRAILING_FIELDS: Partial<Record<WorkflowLibrarySort | ProjectWorkflowSort, WorkflowLibraryListTrailingField>> = {
  created: 'created_at',
  lastRun: 'last_run_at',
  opened: 'opened_at',
};

const PLACE_EMPTY_KEYS: Partial<Record<WorkflowLibraryTab, string>> = {
  recent: 'workflowLibrary.emptyRecent',
  shared: 'workflowLibrary.emptyShared',
  user: 'workflowLibrary.emptyUser',
};

const EMPTY_PROJECT_ENTRIES: ReturnType<typeof toProjectWorkflowLibraryEntries> = [];

/**
 * Only what the shell lays out: the filter (for the toolbar's order), the rail's totals and tag counts. Entries, range
 * windows, enrichment and selected pins belong to `WorkflowLibraryBrowsePanes`, so they never re-render the shell.
 */
const selectShell = (snapshot: WorkflowLibraryBrowseSnapshot) => ({
  filter: snapshot.filter,
  placeTotals: snapshot.placeTotals,
  tagCounts: snapshot.tagCounts,
});

const selectLibraryNavigation = (snapshot: ReturnType<typeof workflowUiStore.getSnapshot>) => ({
  projectOrder: snapshot.libraryProjectOrder,
  query: snapshot.libraryQuery,
  selection: snapshot.librarySelection,
  tab: snapshot.libraryTab,
});

/**
 * Mounted while a template place shows: on opening, or on leaving This project. It applies the opening filter, and on
 * leaving (closing, or switching to This project) drops the search and tags from the listing, so the next opening
 * never paints a filtered view first. The project view never loads library pages.
 */
const WorkflowLibraryBrowseSession = ({
  opening,
  onLeave,
}: {
  opening: WorkflowLibraryBrowseFilter;
  /** Stable; cancels the search still waiting to reach the listing. */
  onLeave: () => void;
}) => {
  useMountEffect(() => {
    // While it holds the listing, a recorded open or run refreshes the view on screen at once.
    const release = holdWorkflowLibraryBrowse();
    void ensureWorkflowLibraryBrowseLoaded(opening);

    return () => {
      release();
      onLeave();
      setWorkflowLibraryBrowseFilter({ search: '', tags: [] });
    };
  });

  return null;
};

/**
 * Once per opening: the rail's place totals, leaving out the place whose unfiltered first page counts it anyway, and,
 * only when the library opened on My library, the switch to Templates for an account with no workflows of its own
 * (unless the user moved or filtered meanwhile).
 */
const WorkflowLibraryOpeningSession = ({
  opening,
  place,
  places,
}: {
  opening: WorkflowLibraryBrowseFilter;
  place: WorkflowLibraryTab;
  places: readonly WorkflowLibraryPlace[];
}) => {
  useMountEffect(() => {
    void loadWorkflowLibraryPlaceTotals(places.filter((candidate) => candidate !== place));

    if (place !== 'user') {
      return;
    }

    void ensureWorkflowLibraryBrowseLoaded(opening).then(() => {
      const { filter, placeTotals } = getWorkflowLibraryBrowseSnapshot();

      if (
        placeTotals.user === 0 &&
        workflowUiStore.getSnapshot().libraryTab === 'user' &&
        filter.place === 'user' &&
        !filter.search &&
        filter.tags.length === 0
      ) {
        setWorkflowLibraryBrowseFilter({ place: 'default', tags: [] });
        setWorkflowLibraryTab('default');
      }
    });
  });

  return null;
};

const SINGLE_USER_PLACES: readonly WorkflowLibraryPlace[] = ['recent', 'user', 'default'];
const MULTIUSER_PLACES: readonly WorkflowLibraryPlace[] = ['recent', 'user', 'default', 'shared'];

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
  const { i18n, t } = useTranslation();
  const tagListFormat = useMemo(
    () => new Intl.ListFormat(i18n.resolvedLanguage, { type: 'disjunction' }),
    [i18n.resolvedLanguage]
  );
  const { filter, placeTotals, tagCounts } = useWorkflowLibraryBrowseSelector(selectShell);
  const { projectOrder, query, selection, tab } = workflowUiStore.useSelector(selectLibraryNavigation);
  const authSession = useAuthSession();
  const projectContext = useOptionalGalleryHostProjectContext();
  const projectId = useWorkflowProjectSelector((project) => project.id);
  const projectWorkflows = useWorkflowProjectSelector((project) => project.workflows);
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const { createWorkflow } = useProjectGraphCommands();
  const { setLibraryPreferences } = useWorkflowUi();
  const sort = useWorkflowPreferencesSelector((preferences) => preferences.workflowLibrarySort);
  const sortDirection = useWorkflowPreferencesSelector((preferences) => preferences.workflowLibrarySortDirection);
  const layout = useWorkflowPreferencesSelector((preferences) => preferences.workflowLibraryLayout);
  const cardSize = useWorkflowPreferencesSelector((preferences) => preferences.workflowLibraryCardSize);
  const thumbnailFit = useWorkflowPreferencesSelector((preferences) => preferences.workflowLibraryThumbnailFit);
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string | null>(null);
  const [removedWorkflowId, setRemovedWorkflowId] = useState<string | null>(null);
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
  const searchInputRef = useRef<HTMLInputElement>(null);
  const placeIdBase = useId();

  const isProjectTab = tab === 'project';
  const places = authSession.multiuserEnabled ? MULTIUSER_PLACES : SINGLE_USER_PLACES;

  const cancelPendingSearch = useCallback(() => {
    if (searchTimerRef.current !== null) {
      clearTimeout(searchTimerRef.current);
      searchTimerRef.current = null;
    }
  }, []);
  const closeDialog = useCallback(() => {
    // Closing the library removes its preview immediately; the parent surface is leaving too. The search it would
    // have applied is cleared with the rest of the query.
    cancelPendingSearch();
    setPreviewRequest(null);
    setIsPreviewOpen(false);
    setContextMenuPoint(null);
    onOpenChange(false);
  }, [cancelPendingSearch, onOpenChange]);
  const { loadPhase, open } = useOpenLibraryWorkflow(closeDialog);
  const isLoadPending = loadPhase !== 'idle';

  // This project filters and orders locally; the template places' results are the server's.
  const projectEntries = useMemo(
    () =>
      isProjectTab
        ? queryProjectWorkflowEntries(
            toProjectWorkflowLibraryEntries(
              projectWorkflows,
              templatesSnapshot.status === 'loaded' ? templatesSnapshot.templates : null
            ),
            { order: projectOrder, search: query.search, tags: query.tags }
          )
        : EMPTY_PROJECT_ENTRIES,
    [isProjectTab, projectOrder, projectWorkflows, query, templatesSnapshot]
  );
  const projectTagCounts = useMemo(() => countProjectWorkflowTags(projectWorkflows), [projectWorkflows]);
  const missingCounts = useWorkflowLibraryMissingCounts(projectEntries);

  // The project selection belongs to the project it was made in; another project starts from its active workflow.
  const projectSelectionId = selection?.projectId === projectId ? selection.workflowId : null;

  // The order the toolbar shows. Before a template place's first listing the store still holds another place's
  // filter, so the account's stored order stands in for it.
  const libraryOrder = getWorkflowLibraryOrder(
    !isProjectTab && filter.place === tab
      ? filter
      : { direction: sortDirection, place: tab === 'project' ? 'user' : tab, sort }
  );
  const shownOrder = isProjectTab ? projectOrder : libraryOrder;
  const trailingField = TRAILING_FIELDS[shownOrder.sort] ?? 'updated_at';
  const directionLabels = useMemo(
    () => ({
      asc: t(getDirectionLabelKey(shownOrder.sort, 'asc')),
      desc: t(getDirectionLabelKey(shownOrder.sort, 'desc')),
    }),
    [shownOrder.sort, t]
  );
  const sortOptions = useMemo<readonly WorkflowLibrarySortOption[]>(() => {
    const keys: readonly (WorkflowLibrarySort | ProjectWorkflowSort)[] =
      tab === 'project' ? PROJECT_SORTS : tab === 'recent' ? ['opened'] : tab === 'default' ? ['name'] : LIBRARY_SORTS;

    return keys.map((key) => ({ label: t(SORT_LABEL_KEYS[key]), value: key }));
  }, [t, tab]);

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
  const getInitialFocus = useCallback(() => searchInputRef.current, []);
  // Escape inside an expanded detail sheet collapses the sheet (its own handler) instead of closing the library.
  const handleEscapeKeyDown = useCallback((event: KeyboardEvent) => {
    if (isWorkflowDetailSheetEscape(event)) {
      event.preventDefault();
    }
  }, []);

  const handlePreviewRequest = useCallback((entry: WorkflowLibraryEntry) => {
    setPreviewRequest({ entry, kind: 'library' });
    setIsPreviewOpen(true);
  }, []);
  // A card's Preview quick action names an id; the freshest resident or selected revision answers it.
  const handleCardPreview = useCallback(
    (workflowId: string) => {
      const entry = selectWorkflowLibraryEntry(getWorkflowLibraryBrowseSnapshot(), workflowId);

      if (entry) {
        handlePreviewRequest(entry);
      }
    },
    [handlePreviewRequest]
  );
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

  const handleSearchChange = useCallback(
    (value: string) => {
      setWorkflowLibrarySearch(value);
      cancelPendingSearch();

      // This project filters as you type; template places ask the server once typing pauses.
      if (!isProjectTab) {
        searchTimerRef.current = setTimeout(() => {
          searchTimerRef.current = null;
          setWorkflowLibraryBrowseFilter({ search: value });
        }, SEARCH_DEBOUNCE_MS);
      }
    },
    [cancelPendingSearch, isProjectTab]
  );

  const handlePlaceChange = useCallback(
    (place: WorkflowLibraryTab) => {
      setContextMenuPoint(null);
      cancelPendingSearch();
      setWorkflowLibraryTab(place);

      // Before the first template listing, the browse session applies the opening filter when it mounts.
      if (place !== 'project' && getWorkflowLibraryBrowseSnapshot().status !== 'idle') {
        setWorkflowLibraryBrowseFilter({ direction: sortDirection, place, search: query.search, sort, tags: [] });
      }
    },
    [cancelPendingSearch, query.search, sort, sortDirection]
  );

  const handleTagsChange = useCallback(
    (tags: readonly string[]) => {
      setWorkflowLibraryTags(tags);

      if (!isProjectTab) {
        setWorkflowLibraryBrowseFilter({ tags });
      }
    },
    [isProjectTab]
  );

  const handleSortChange = useCallback(
    (value: string) => {
      if (isProjectTab) {
        const next = PROJECT_SORTS.find((key) => key === value);

        if (next) {
          setWorkflowLibraryProjectOrder({
            direction: next === 'order' || next === 'name' ? 'asc' : 'desc',
            sort: next,
          });
        }
        return;
      }

      const next = LIBRARY_SORTS.find((key) => key === value);

      // Only My library and Shared offer a choice; a new key starts in its natural direction.
      if (next && (tab === 'user' || tab === 'shared')) {
        const direction = getWorkflowLibrarySortDefaultDirection(next);

        setLibraryPreferences({ workflowLibrarySort: next, workflowLibrarySortDirection: direction });
        setWorkflowLibraryBrowseFilter({ direction, sort: next });
      }
    },
    [isProjectTab, setLibraryPreferences, tab]
  );

  const handleDirectionToggle = useCallback(() => {
    if (isProjectTab) {
      setWorkflowLibraryProjectOrder({ ...projectOrder, direction: flipDirection(projectOrder.direction) });
      return;
    }

    const direction = flipDirection(libraryOrder.direction);

    if (tab === 'default') {
      // Templates always list by name; reversing them lasts until another place applies the account's order.
      setWorkflowLibraryBrowseFilter({ direction, sort: 'name' });
    } else if (tab === 'user' || tab === 'shared') {
      setLibraryPreferences({ workflowLibrarySortDirection: direction });
      setWorkflowLibraryBrowseFilter({ direction });
    }
  }, [isProjectTab, libraryOrder.direction, projectOrder, setLibraryPreferences, tab]);

  const handleLayoutChange = useCallback(
    (workflowLibraryLayout: WorkflowLibraryLayout) => setLibraryPreferences({ workflowLibraryLayout }),
    [setLibraryPreferences]
  );
  const handleCardSizeChange = useCallback(
    (workflowLibraryCardSize: WorkflowLibraryCardSize) => setLibraryPreferences({ workflowLibraryCardSize }),
    [setLibraryPreferences]
  );
  const handleThumbnailFitChange = useCallback(
    (workflowLibraryThumbnailFit: WorkflowThumbnailFit) => setLibraryPreferences({ workflowLibraryThumbnailFit }),
    [setLibraryPreferences]
  );
  const handleNewWorkflow = useCallback(
    () => setWorkflowLibrarySelection({ projectId, workflowId: createWorkflow() }),
    [createWorkflow, projectId]
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

  const handleDeleted = useCallback((workflowId: string | null) => {
    setRemovedWorkflowId(workflowId);
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

  // Read once, when a session mounts.
  const openingFilter = useMemo<WorkflowLibraryBrowseFilter>(
    () => ({
      direction: sortDirection,
      place: tab === 'project' ? 'user' : tab,
      search: query.search,
      sort,
      tags: query.tags,
    }),
    [query, sort, sortDirection, tab]
  );

  // The result line counts what the grid can show: every match on This project, the server's total elsewhere. The
  // missing-model count is exact only while every match is resident, so a partial window shows none.
  const tagsText = query.tags.length > 0 ? tagListFormat.format(query.tags) : '';
  // With no search or tags, an empty place says why, rather than that nothing matched.
  const emptyKey = query.search || query.tags.length > 0 ? undefined : PLACE_EMPTY_KEYS[tab];
  const emptyMessage = emptyKey ? t(emptyKey) : undefined;

  return (
    <>
      <Dialog.Root
        initialFocusEl={getInitialFocus}
        open={isOpen}
        size="xl"
        onEscapeKeyDown={handleEscapeKeyDown}
        onOpenChange={handleDialogOpenChange}
      >
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content
              ref={preloadGraphPreview}
              css={CONTENT_CSS}
              aria-busy={isLoadPending}
              h="80vh"
              maxH="80vh"
              maxW="min(80rem, calc(100vw - 4rem))"
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
              <Dialog.Header
                alignItems="center"
                borderBottomWidth="1px"
                borderColor="border.subtle"
                display="flex"
                flex="none"
                gap="3"
                h="12"
                px="4"
                py="0"
              >
                <HStack align="baseline" flex="1" gap="2.5" minW="0">
                  <Dialog.Title flexShrink={0}>{t('workflowLibrary.title')}</Dialog.Title>
                  {projectContext?.projectName ? (
                    <Text color="fg.muted" fontSize="md" minW="0" truncate>
                      {projectContext.projectName}
                    </Text>
                  ) : null}
                </HStack>
                <Dialog.CloseTrigger asChild>
                  <CloseButton disabled={isLoadPending} insetEnd="auto" position="static" top="auto" />
                </Dialog.CloseTrigger>
              </Dialog.Header>
              <Dialog.Body
                css={BODY_CSS}
                data-library-tab={tab}
                data-pending-preview={previewGraphId || undefined}
                flex="1"
                gap="0"
                minH="0"
                p="0"
              >
                <Box css={RAIL_SLOT_CSS} display="flex" minH="0">
                  <WorkflowLibraryRail
                    checkedTags={query.tags}
                    footer={isProjectTab ? t('workflowLibrary.thisProjectHint') : undefined}
                    idBase={placeIdBase}
                    place={tab}
                    placeTotals={placeTotals}
                    projectTotal={projectWorkflows.length}
                    showShared={authSession.multiuserEnabled}
                    tagCounts={isProjectTab ? projectTagCounts : tagCounts}
                    onPlaceChange={handlePlaceChange}
                    onTagsChange={handleTagsChange}
                  />
                </Box>
                <Box as="section" aria-labelledby={getWorkflowLibraryPlaceLabelId(placeIdBase, tab)} css={CENTRE_CSS}>
                  <WorkflowLibraryToolbar
                    narrowLead={
                      <>
                        <WorkflowLibraryPlaceSelect
                          place={tab}
                          placeTotals={placeTotals}
                          projectTotal={projectWorkflows.length}
                          showShared={authSession.multiuserEnabled}
                          onPlaceChange={handlePlaceChange}
                        />
                        <WorkflowLibraryTagButton
                          checkedTags={query.tags}
                          getFallbackFocus={getInitialFocus}
                          tagCounts={isProjectTab ? projectTagCounts : tagCounts}
                          onTagsChange={handleTagsChange}
                        />
                      </>
                    }
                    cardSize={cardSize}
                    direction={shownOrder.direction}
                    directionLabels={directionLabels}
                    isDirectionFixed={tab === 'recent'}
                    isSortFixed={tab === 'recent' || tab === 'default'}
                    layout={layout}
                    search={query.search}
                    searchInputRef={searchInputRef}
                    sort={shownOrder.sort}
                    sortOptions={sortOptions}
                    thumbnailFit={thumbnailFit}
                    onCardSizeChange={handleCardSizeChange}
                    onDirectionToggle={handleDirectionToggle}
                    onLayoutChange={handleLayoutChange}
                    onNewWorkflow={isProjectTab ? handleNewWorkflow : undefined}
                    onSearchChange={handleSearchChange}
                    onSortChange={handleSortChange}
                    onThumbnailFitChange={handleThumbnailFitChange}
                  />
                  <WorkflowLibraryTagChips
                    checkedTags={query.tags}
                    getFallbackFocus={getInitialFocus}
                    onTagsChange={handleTagsChange}
                  />
                  <WorkflowThumbnailFitContext value={thumbnailFit}>
                    <WorkflowLibraryCardSizeContext value={cardSize}>
                      {isProjectTab ? (
                        <>
                          <WorkflowLibraryResultLine count={projectEntries.length} tagsText={tagsText} />
                          <ProjectWorkflowsView
                            contextMenuPoint={contextMenuPoint}
                            contextMenuTriggerId={contextMenuTriggerId}
                            entries={projectEntries}
                            layout={layout}
                            missingCounts={missingCounts}
                            selectedWorkflowId={projectSelectionId}
                            trailingField={trailingField}
                            onClose={closeDialog}
                            onContextMenu={handleCardContextMenu}
                            onContextMenuClose={closeContextMenu}
                            onPreview={handleProjectPreviewRequest}
                            onSelect={handleProjectSelect}
                          />
                        </>
                      ) : (
                        <WorkflowLibraryBrowsePanes
                          contextMenuPoint={contextMenuPoint}
                          contextMenuTriggerId={contextMenuTriggerId}
                          emptyMessage={emptyMessage}
                          isOpen={isOpen}
                          layout={layout}
                          projectWorkflows={projectWorkflows}
                          removedWorkflowId={removedWorkflowId}
                          selectedWorkflowId={selectedWorkflowId}
                          tab={tab}
                          tagsText={tagsText}
                          trailingField={trailingField}
                          onCardPreview={handleCardPreview}
                          onClose={closeDialog}
                          onContextMenu={handleCardContextMenu}
                          onContextMenuClose={closeContextMenu}
                          onDeleted={handleDeleted}
                          onOpen={handleOpenWorkflow}
                          onOpenItem={handleOpenItem}
                          onPreview={handlePreviewRequest}
                          onSelect={setSelectedWorkflowId}
                        />
                      )}
                    </WorkflowLibraryCardSizeContext>
                  </WorkflowThumbnailFitContext>
                </Box>
              </Dialog.Body>
              {isOpen ? <WorkflowLibraryOpeningSession opening={openingFilter} place={tab} places={places} /> : null}
              {isOpen && !isProjectTab ? (
                <WorkflowLibraryBrowseSession opening={openingFilter} onLeave={cancelPendingSearch} />
              ) : null}
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
