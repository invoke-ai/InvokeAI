import type {
  WorkflowLibraryBrowseFilter,
  WorkflowLibraryBrowseSnapshot,
  WorkflowLibraryPlace,
  WorkflowLibrarySort,
} from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowLibraryListItem } from '@features/workflow/queries';
import type { WorkflowPreviewView } from '@features/workflow/ui/graph-preview/WorkflowPreviewStage';
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
import { WORKFLOW_PREVIEW_DIALOG_SIZE } from '@features/workflow/ui/graph-preview/WorkflowPreviewStage';
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
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';

import type { WorkflowLibraryPanesPreview } from './WorkflowLibraryBrowsePanes';
import type { WorkflowCardMenuAnchor } from './WorkflowLibraryCard';
import type { WorkflowContextMenuTarget } from './WorkflowLibraryContextMenu';
import type { WorkflowLibraryListTrailingField } from './WorkflowLibraryList';
import type { WorkflowLibraryPreviewStageHandle } from './WorkflowLibraryPreviewStage';
import type { WorkflowLibraryCardSize } from './workflowThumbnail';

import { preloadGraphPreview } from './libraryPreviewSource';
import {
  countProjectWorkflowTags,
  queryProjectWorkflowEntries,
  toProjectWorkflowLibraryEntries,
} from './projectWorkflowEntries';
import { ProjectWorkflowsView } from './ProjectWorkflowsView';
import { planLibraryWorkflowOpen, useOpenLibraryWorkflow } from './useOpenLibraryWorkflow';
import { isWorkflowDetailSheetEscape } from './WorkflowDetailSheet';
import { WorkflowLibraryBrowsePanes, WorkflowLibraryResultLine } from './WorkflowLibraryBrowsePanes';
import { getWorkflowLibraryCardId } from './WorkflowLibraryCard';
import {
  getWorkflowLibraryPlaceLabelId,
  getWorkflowLibraryPlaceLabelKey,
  WORKFLOW_LIBRARY_NARROW,
  WORKFLOW_LIBRARY_WIDE,
  WorkflowLibraryPlaceSelect,
  WorkflowLibraryRail,
  WorkflowLibraryTagButton,
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
 * Wide: one body grid with the rails as full-height columns; the centre and panes boxes are `display: contents`, so
 * their parts are grid items. The body is the `workflow-library-content` container and scrolls when too short.
 */
const BODY_CSS: SystemStyleObject = {
  display: 'flex',
  // Preview: the stage covers the rail and collection, which stay laid out but leave the tab order and a11y tree.
  '&[data-previewing] > [data-library-rail-slot], &[data-previewing] [data-workflow-library-toolbar], &[data-previewing] [data-workflow-library-results], &[data-previewing] [data-workflow-library-collection]':
    { visibility: 'hidden' },
  [WORKFLOW_LIBRARY_NARROW]: {
    '&[data-previewing] [data-workflow-library-toolbar], &[data-previewing] [data-workflow-library-results]': {
      display: 'none',
    },
  },
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
/** Narrow: the centre column is the cards' and sheet's container, and scrolls when they cannot share its height. */
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
  // When the column scrolls (a flow layout), scrolling into view clears its padding and the sheet's sticky summary.
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

/** Shell fields only: entries, windows and enrichment stay in `WorkflowLibraryBrowsePanes`, off the shell's renders. */
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

/** Applies the opening filter; on leaving, drops search and tags so the next opening never paints them first. */
const WorkflowLibraryBrowseSession = ({
  opening,
  onLeave,
}: {
  opening: WorkflowLibraryBrowseFilter;
  /** Stable; cancels the search still waiting to reach the listing. */
  onLeave: () => void;
}) => {
  useMountEffect(() => {
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

/** Once per opening: rail totals, and on My library the switch to Templates for an account with no workflows. */
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
  /** The opened workflow and its opener (focus returns there); cleared on close, or the shell would reopen on it. */
  const [preview, setPreview] = useState<{ opener: HTMLElement | null; workflowId: string } | null>(null);
  const [previewView, setPreviewView] = useState<WorkflowPreviewView>('nodes');
  // However the library closes (its own controls, the copy question, the host), it reopens on the collection, on Graph.
  if (!isOpen && (preview !== null || previewView !== 'nodes')) {
    setPreview(null);
    setPreviewView('nodes');
  }
  const contentRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<WorkflowLibraryPreviewStageHandle | null>(null);
  // The target outlives the anchor, for focus return and the dialogs its items open.
  const [contextMenuAnchor, setContextMenuAnchor] = useState<WorkflowCardMenuAnchor | null>(null);
  const [contextMenuTarget, setContextMenuTarget] = useState<WorkflowContextMenuTarget | null>(null);
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
    cancelPendingSearch();
    setContextMenuAnchor(null);
    onOpenChange(false);
  }, [cancelPendingSearch, onOpenChange]);
  const { loadPhase, open } = useOpenLibraryWorkflow(closeDialog);
  const isLoadPending = loadPhase !== 'idle';

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

  // A menu whose workflow left the view has nothing to act on: close it.
  const contextMenuWorkflowId = contextMenuAnchor?.workflowId ?? null;
  const isLibraryMenuTargetShown = useWorkflowLibraryBrowseSelector(
    (snapshot) =>
      contextMenuWorkflowId === null ||
      isProjectTab ||
      selectWorkflowLibraryEntry(snapshot, contextMenuWorkflowId) !== undefined
  );
  if (
    contextMenuWorkflowId !== null &&
    !(isProjectTab
      ? projectEntries.some((entry) => entry.item.workflow_id === contextMenuWorkflowId)
      : isLibraryMenuTargetShown)
  ) {
    setContextMenuAnchor(null);
  }
  const missingCounts = useWorkflowLibraryMissingCounts(projectEntries);

  // The project selection belongs to the project it was made in; another project starts from its active workflow.
  const projectSelectionId = selection?.projectId === projectId ? selection.workflowId : null;

  // Before a template place's first listing the store holds another place's filter, so the stored order stands in.
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
  // Escape collapses an expanded sheet, else closes a node card, else leaves preview; only then closes the library.
  const handleEscapeKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // StrictMode registers the dialog's layer twice; the first Escape acts and its preventDefault stops the next.
      if (event.defaultPrevented) {
        return;
      }

      if (isWorkflowDetailSheetEscape(event)) {
        event.preventDefault();
      } else if (preview) {
        event.preventDefault();
        stageRef.current?.handleEscape();
      }
    },
    [preview]
  );
  const setContentRef = useCallback((element: HTMLDivElement | null) => {
    contentRef.current = element;

    if (element) {
      preloadGraphPreview();
    }
  }, []);

  // The stage takes focus from the asking control, which hiding the collection would otherwise drop to the document.
  const enterPreview = useCallback(
    (workflowId: string, view?: WorkflowPreviewView) => {
      const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;

      flushSync(() => {
        if (view) {
          setPreviewView(view);
        }

        if (isProjectTab) {
          setWorkflowLibrarySelection({ projectId, workflowId });
        } else {
          setSelectedWorkflowId(workflowId);
        }

        setContextMenuAnchor(null);
        setPreview({ opener, workflowId });
      });
      contentRef.current?.querySelector<HTMLElement>('[data-preview-stage]')?.focus({ preventScroll: true });
    },
    [isProjectTab, projectId]
  );
  // Focus returns to the opener while it shows that workflow; after stepping, to the selected card or row.
  const leavePreview = useCallback(
    (workflowId: string | null) => {
      const opener = preview?.opener ?? null;
      const isRendered = (element: HTMLElement | null | undefined): element is HTMLElement =>
        element?.isConnected === true && element.getClientRects().length > 0;

      flushSync(() => setPreview(null));

      const selected = workflowId ? document.getElementById(getWorkflowLibraryCardId(workflowId)) : null;
      const target =
        workflowId === preview?.workflowId && isRendered(opener)
          ? opener
          : isRendered(selected)
            ? selected
            : isRendered(opener)
              ? opener
              : searchInputRef.current;

      target?.focus();
    },
    [preview]
  );
  const panesPreview = useMemo<WorkflowLibraryPanesPreview>(
    () => ({
      isActive: preview !== null,
      onLeave: leavePreview,
      onViewChange: setPreviewView,
      placeLabel: t(getWorkflowLibraryPlaceLabelKey(tab)),
      view: previewView,
    }),
    [leavePreview, preview, previewView, t, tab]
  );

  const handleSearchChange = useCallback(
    (value: string) => {
      setWorkflowLibrarySearch(value);
      cancelPendingSearch();

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
      setContextMenuAnchor(null);
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
      const entry = selectWorkflowLibraryEntry(getWorkflowLibraryBrowseSnapshot(), workflowId);

      if (entry) {
        handleOpenItem(entry.item);
      }
    },
    [handleOpenItem]
  );

  const handleDeleted = useCallback((workflowId: string) => {
    setRemovedWorkflowId(workflowId);
    setSelectedWorkflowId((current) => (current === workflowId ? null : current));
    setContextMenuAnchor(null);
  }, []);
  const handleCardContextMenu = useCallback(
    (workflowId: string, anchor: WorkflowCardMenuAnchor) => {
      const target = { triggerId: anchor.kind === 'trigger' ? anchor.triggerId : null, workflowId };

      if (
        anchor.kind === 'trigger' &&
        contextMenuAnchor?.kind === 'trigger' &&
        contextMenuAnchor.triggerId === anchor.triggerId
      ) {
        setContextMenuAnchor(null);
        return;
      }

      // An open menu does not follow a new anchor: close it, then reopen at the new point once that close renders.
      if (contextMenuAnchor) {
        setContextMenuAnchor(null);
        requestAnimationFrame(() => {
          setContextMenuTarget(target);
          setContextMenuAnchor(anchor);
        });
      } else {
        setContextMenuTarget(target);
        setContextMenuAnchor(anchor);
      }
    },
    [contextMenuAnchor]
  );
  const closeContextMenu = useCallback(() => setContextMenuAnchor(null), []);
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

  // The missing-model count is exact only while every match is resident, so a partial window shows none.
  const tagsText = query.tags.length > 0 ? tagListFormat.format(query.tags) : '';
  const emptyKey = query.search || query.tags.length > 0 ? undefined : PLACE_EMPTY_KEYS[tab];
  const emptyMessage = emptyKey ? t(emptyKey) : undefined;

  return (
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
            ref={setContentRef}
            css={CONTENT_CSS}
            aria-busy={isLoadPending}
            {...WORKFLOW_PREVIEW_DIALOG_SIZE}
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
              data-previewing={preview ? '' : undefined}
              flex="1"
              gap="0"
              minH="0"
              p="0"
            >
              <Box css={RAIL_SLOT_CSS} data-library-rail-slot="" display="flex" minH="0">
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
                <WorkflowThumbnailFitContext value={thumbnailFit}>
                  <WorkflowLibraryCardSizeContext value={cardSize}>
                    {isProjectTab ? (
                      <>
                        <WorkflowLibraryResultLine count={projectEntries.length} tagsText={tagsText} />
                        <ProjectWorkflowsView
                          contextMenuAnchor={contextMenuAnchor}
                          contextMenuTarget={contextMenuTarget}
                          entries={projectEntries}
                          layout={layout}
                          missingCounts={missingCounts}
                          selectedWorkflowId={projectSelectionId}
                          trailingField={trailingField}
                          onClose={closeDialog}
                          onContextMenu={handleCardContextMenu}
                          onContextMenuClose={closeContextMenu}
                          onPreview={enterPreview}
                          onSelect={handleProjectSelect}
                          preview={panesPreview}
                          previewStageRef={stageRef}
                        />
                      </>
                    ) : (
                      <WorkflowLibraryBrowsePanes
                        contextMenuAnchor={contextMenuAnchor}
                        contextMenuTarget={contextMenuTarget}
                        emptyMessage={emptyMessage}
                        isOpen={isOpen}
                        layout={layout}
                        projectWorkflows={projectWorkflows}
                        removedWorkflowId={removedWorkflowId}
                        selectedWorkflowId={selectedWorkflowId}
                        tab={tab}
                        tagsText={tagsText}
                        trailingField={trailingField}
                        preview={panesPreview}
                        previewStageRef={stageRef}
                        onClose={closeDialog}
                        onContextMenu={handleCardContextMenu}
                        onContextMenuClose={closeContextMenu}
                        onDeleted={handleDeleted}
                        onOpen={handleOpenWorkflow}
                        onOpenItem={handleOpenItem}
                        onPreview={enterPreview}
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
  );
};
