// ListItem takes its slots as JSX, and the List calls renderItem afresh on each render; rows are what is memoised.
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';
import type { ListRange, ListRowProps } from '@platform/ui/list/List';
import type { ListContextMenuAnchor } from '@platform/ui/list/ListItem';
import type { ListRow } from '@platform/ui/list/listRows';
import type { FocusEvent, MouseEvent, UIEvent } from 'react';

import { Badge, Box, HStack, Icon, type SystemStyleObject, Text } from '@chakra-ui/react';
import {
  getWorkflowLibraryBrowseSnapshot,
  loadWorkflowLibraryRange,
  refreshWorkflowLibraryBrowse,
  rememberWorkflowLibraryScroll,
} from '@features/workflow/data/libraryBrowseStore';
import { useInvocationTemplatesSnapshot } from '@features/workflow/react';
import { Button, IconButton } from '@platform/ui/Button';
import { getListContentHeight, List } from '@platform/ui/list/List';
import { ListItem } from '@platform/ui/list/ListItem';
import { Tooltip } from '@platform/ui/Tooltip';
import { EllipsisIcon, FolderCheckIcon, WorkflowIcon } from 'lucide-react';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowLibraryGridProps } from './WorkflowLibraryGrid';

import { formatTimeSince, useDataTimeSnapshot } from './relativeTime';
import { WORKFLOW_LIBRARY_FLOW_LAYOUT, WorkflowSummaryThumbnail } from './WorkflowDetailSheet';
import {
  getWorkflowLibraryCardId,
  getWorkflowLibraryCardMenuContentId,
  getWorkflowLibraryCardMenuId,
  type WorkflowCardMenuAnchor,
} from './WorkflowLibraryCard';
import {
  dedupeWorkflowLibraryEntries,
  getWorkflowLibraryDemandIds,
  useFrozenWhileHidden,
  useLinkedLibraryTemplates,
  WorkflowLibraryDemand,
} from './workflowLibraryCollection';
import { getWorkflowLibraryEntryFacts, WorkflowLibraryFactsLine } from './WorkflowLibraryFacts';
import { getMissingNodePacks } from './WorkflowRequirementsList';

/**
 * The library's list layout: the platform List over the same collection the grid shows. A templates place lists every
 * server position, rows outside the resident window as placeholders, and loads the window around the List's visible
 * range through the shared demand module; the This-project view lists its own entries. Rows select on press, open on
 * double-click, and raise the same actions menu as a card from a right-click, Shift+F10 or their More button.
 */

/** The date column a row shows, chosen by the caller from the current order; null shows none. */
export type WorkflowLibraryListTrailingField = 'updated_at' | 'opened_at' | 'created_at' | 'last_run_at' | null;

export interface WorkflowLibraryListProps extends WorkflowLibraryGridProps {
  trailingField?: WorkflowLibraryListTrailingField;
  isActive?: boolean;
}

interface WorkflowLibraryListItem {
  /** Null while the position's metadata is not resident. */
  entry: WorkflowLibraryEntry | null;
  index: number;
}

const ROW_PITCH = getListContentHeight([{ kind: 'item' }], 'comfortable');
/** Rows loaded past each edge of the visible range; the List mounts as many, so they show data once scrolled to. */
const DEMAND_OVERSCAN_ROWS = 8;
const NO_MISSING_COUNTS: ReadonlyMap<string, number> = new Map();
/**
 * Row controls take the row's text colour and a border and hover tint mixed from it, so they read on the selected
 * row's accent fill as on the panel, without a fill of their own.
 */
const ROW_ACTION_CSS: SystemStyleObject = {
  borderColor: 'color-mix(in oklab, currentColor 35%, transparent)',
  color: 'inherit',
  _hover: { bg: 'color-mix(in oklab, currentColor 12%, transparent)' },
  _expanded: { bg: 'color-mix(in oklab, currentColor 12%, transparent)' },
};
/**
 * A position not yet resident: a thumbnail well and a line in a translucent tint of the text colour, static so fast
 * scrolls paint nothing, and never a solid block on any row surface.
 */
const PLACEHOLDER_THUMBNAIL_PROPS = {
  aspectRatio: 3 / 2,
  bg: 'fg/10',
  flexShrink: 0,
  rounded: 'sm',
  w: '3.375rem',
} as const;
const PLACEHOLDER_LINE_PROPS = { bg: 'fg/10', display: 'block', h: '2', rounded: 'full', w: '8rem' } as const;

/**
 * Open and Preview are always mounted and keep their width, so rows never reflow under the pointer; they show while the
 * pointer is over the row, focus is inside it or its menu is open. One that does not apply (Open on the active project
 * workflow, both on a placeholder) stays hidden. Without hover (touch) they never show and More, always visible,
 * carries the row's actions.
 */
const LIST_CSS: SystemStyleObject = {
  display: 'flex',
  flex: '1',
  flexDirection: 'column',
  gap: '2',
  minH: '0',
  minW: '0',
  '& [data-row-quick-action]': { visibility: 'hidden' },
  '& [data-list-row]:focus-within [data-row-quick-action]:not([data-unavailable]), & [data-list-row][data-menu-open] [data-row-quick-action]:not([data-unavailable])':
    { visibility: 'visible' },
  '@media (hover: hover)': {
    '& [data-list-row]:hover [data-row-quick-action]:not([data-unavailable])': { visibility: 'visible' },
  },
  '@media (hover: none)': { '& [data-row-quick-action]': { display: 'none' } },
  // The scrolling column gives the list no height of its own; without one it would mount every row.
  [WORKFLOW_LIBRARY_FLOW_LAYOUT]: { flex: '1 1 16rem', height: '16rem' },
};

const toScrollAnchor = (offset: number) => {
  const index = Math.max(0, Math.floor(offset / ROW_PITCH));

  return { fraction: Math.max(0, Math.min(1, (offset - index * ROW_PITCH) / ROW_PITCH)), index };
};

const rowOf = (target: EventTarget) =>
  target instanceof Element ? target.closest<HTMLElement>('[data-list-row]') : null;

/** A row's server position: rows carry their 1-based set position, and the list has no section headers. */
const positionOf = (row: HTMLElement) => Number(row.getAttribute('aria-posinset')) - 1;

interface KeptRow {
  entry: WorkflowLibraryEntry;
  index: number;
}

const isKept = (kept: KeptRow | null, entry: WorkflowLibraryEntry, index: number): boolean =>
  kept?.index === index &&
  kept.entry.item.workflow_id === entry.item.workflow_id &&
  kept.entry.item.revision === entry.item.revision;

const keep = (entry: WorkflowLibraryEntry, index: number): KeptRow => ({
  entry: { ...entry, enrichment: { status: 'pending' } },
  index,
});

const entryAt = (rows: readonly ListRow<WorkflowLibraryListItem>[], index: number): WorkflowLibraryEntry | null => {
  const row = rows[index];

  return row?.kind === 'item' ? row.item.entry : null;
};

/** A windowed row's key: its server position, whether or not its metadata is resident. */
const positionKey = (index: number) => `position:${index}`;

export const WorkflowLibraryList = ({
  activeWorkflowId = null,
  emptyMessage,
  entries,
  error,
  isHidden = false,
  isActive = true,
  missingCounts = NO_MISSING_COUNTS,
  openMenuAnchor = null,
  removedWorkflowId = null,
  selectedWorkflowId,
  showVisibilityStatus = false,
  status,
  trailingField = 'updated_at',
  window: browseWindow,
  onContextMenu,
  onOpen,
  onPreview,
  onSelect,
}: WorkflowLibraryListProps) => {
  const { t } = useTranslation();
  const wrapperRef = useRef<HTMLDivElement>(null);
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const templates = templatesSnapshot.status === 'loaded' ? templatesSnapshot.templates : null;
  const isWindowed = browseWindow !== undefined;
  // Only library rows mark templates the project holds; the project view is that collection.
  const linkedTemplates = useLinkedLibraryTemplates(!isWindowed);
  const filter = browseWindow?.filter;
  const visibleEntries = useMemo(() => dedupeWorkflowLibraryEntries(entries), [entries]);
  const startIndex = browseWindow?.startIndex ?? 0;
  const total = browseWindow?.total ?? visibleEntries.length;

  /**
   * Rows kept while their page is not resident, with their positions: the one last focused or given a menu, and the
   * selected one. Each stays rendered with its name, so focus and the List's tab stop never land on a placeholder.
   * The copies drop their parsed documents, like the grid's pinned card, so memory stays within the window.
   */
  const [focusedRow, setFocusedRow] = useState<KeptRow | null>(null);
  const [selectedRow, setSelectedRow] = useState<KeptRow | null>(null);
  if (focusedRow !== null && focusedRow.entry.item.workflow_id === removedWorkflowId) {
    setFocusedRow(null);
  }
  if (selectedRow !== null && selectedRow.entry.item.workflow_id === removedWorkflowId) {
    setSelectedRow(null);
  }
  // The selection can change from outside (the rail, the store's first-entry fallback): record it while resident.
  const selectedResidentIndex =
    isWindowed && selectedWorkflowId !== null
      ? visibleEntries.findIndex((entry) => entry.item.workflow_id === selectedWorkflowId)
      : -1;
  const selectedResident = visibleEntries[selectedResidentIndex];
  if (
    selectedResident &&
    selectedResident.item.workflow_id !== removedWorkflowId &&
    !isKept(selectedRow, selectedResident, startIndex + selectedResidentIndex)
  ) {
    setSelectedRow(keep(selectedResident, startIndex + selectedResidentIndex));
  }
  const retain = useCallback(
    (entry: WorkflowLibraryEntry, index: number) => {
      if (isWindowed) {
        setFocusedRow((current) => (isKept(current, entry, index) ? current : keep(entry, index)));
      }
    },
    [isWindowed]
  );

  const rows = useMemo((): ListRow<WorkflowLibraryListItem>[] => {
    if (!isWindowed) {
      return visibleEntries.map((entry, index) => ({
        item: { entry, index },
        key: entry.item.workflow_id,
        kind: 'item',
      }));
    }

    // A kept row whose workflow is resident elsewhere (the order changed) would duplicate its key.
    const residentIds = new Set(visibleEntries.map((entry) => entry.item.workflow_id));
    const keptRows = new Map<number, WorkflowLibraryEntry>();
    for (const kept of [selectedRow, focusedRow]) {
      if (kept !== null && kept.index < total && !residentIds.has(kept.entry.item.workflow_id)) {
        residentIds.add(kept.entry.item.workflow_id);
        keptRows.set(kept.index, kept.entry);
      }
    }

    // Keyed by position, so a placeholder and the entry that replaces it are one row: its button, and focus on it,
    // survive the page arriving.
    return Array.from({ length: total }, (_, index): ListRow<WorkflowLibraryListItem> => ({
      item: { entry: visibleEntries[index - startIndex] ?? keptRows.get(index) ?? null, index },
      key: positionKey(index),
      kind: 'item',
    }));
  }, [focusedRow, isWindowed, selectedRow, startIndex, total, visibleEntries]);

  // The selected row's key: the position showing the selected workflow, resident or kept. A kept position another
  // workflow now holds (the server reordered while the selection was out of the window) marks nothing, so the List's
  // tab stop falls back rather than highlight the wrong row.
  const selectedIndex =
    selectedResident !== undefined
      ? startIndex + selectedResidentIndex
      : selectedRow !== null && selectedRow.entry.item.workflow_id === selectedWorkflowId
        ? selectedRow.index
        : -1;
  const activeKey = !isWindowed
    ? selectedWorkflowId
    : selectedIndex >= 0 && entryAt(rows, selectedIndex)?.item.workflow_id === selectedWorkflowId
      ? positionKey(selectedIndex)
      : null;

  // The visible range the List reports; null until its viewport has measured.
  const [range, setRange] = useState<ListRange | null>(null);
  const rangeReports = useRef(0);
  const handleRangeChange = useCallback(
    (next: ListRange) => {
      const viewport = wrapperRef.current?.querySelector<HTMLElement>('[data-list-viewport]');
      if (!isActive || (viewport && viewport.clientHeight === 0)) {
        return;
      }
      setRange(next);
      rangeReports.current += 1;

      // The first report is the restored position itself; later ones keep the view's anchor current, so leaving for
      // another filter (which unmounts this list after the store has moved on) still returns here.
      if (filter && viewport && rangeReports.current > 1) {
        rememberWorkflowLibraryScroll(filter, toScrollAnchor(viewport.scrollTop));
      }
    },
    [filter, isActive]
  );
  // Passive unmount cleanup runs after DOM refs detach; retain only offsets observed while the list is visible.
  const visibleScrollOffset = useRef<number | null>(null);
  const handleScrollCapture = useCallback(
    (event: UIEvent<HTMLDivElement>) => {
      const viewport = event.target as HTMLElement;
      if (isActive && viewport.matches('[data-list-viewport]') && viewport.clientHeight > 0) {
        visibleScrollOffset.current = viewport.scrollTop;
      }
    },
    [isActive]
  );
  const persistScrollOffset = useCallback(() => {
    if (filter && visibleScrollOffset.current !== null) {
      rememberWorkflowLibraryScroll(filter, toScrollAnchor(visibleScrollOffset.current));
    }
  }, [filter]);
  // Shares the grid's anchor for this filter; its position converts to a row offset, close enough across layouts.
  const [initialScrollOffset] = useState(() => {
    const snapshot = getWorkflowLibraryBrowseSnapshot();
    const anchor = filter !== undefined && filter === snapshot.filter ? snapshot.scrollAnchor : null;

    return anchor ? (anchor.index + anchor.fraction) * ROW_PITCH : undefined;
  });

  const visibleFirst = range?.startIndex ?? 0;
  const visibleLast = Math.min(total - 1, range?.endIndex ?? 0);
  const first = Math.max(0, visibleFirst - DEMAND_OVERSCAN_ROWS);
  const last = Math.min(total - 1, visibleLast + DEMAND_OVERSCAN_ROWS);
  const ids = useMemo(
    () => getWorkflowLibraryDemandIds(visibleEntries, startIndex, { first, last, visibleFirst, visibleLast }),
    [visibleEntries, startIndex, first, last, visibleFirst, visibleLast]
  );
  const demandKey = useFrozenWhileHidden(`${first}:${last}:${ids.join(',')}`, isHidden);
  const hasEntries = visibleEntries.length > 0;
  // Idle precedes the first open-time load; show loading rather than an empty result.
  const isPending = status === 'idle' || status === 'loading';
  const listStatus = !hasEntries && isPending ? 'loading' : !hasEntries && error ? 'error' : 'ready';

  const handleRetry = useCallback(() => {
    void loadWorkflowLibraryRange(first, last, true);
  }, [first, last]);
  // With nothing resident there is no window to reread: the first listing itself failed. The button stays busy until
  // that listing settles.
  const [isReloading, setIsReloading] = useState(false);
  const handleReload = useCallback(() => {
    setIsReloading(true);
    void refreshWorkflowLibraryBrowse().finally(() => setIsReloading(false));
  }, []);

  /** A row acted on outside the resident window (a retained one) reloads its position first, as a pinned card does. */
  const ensurePosition = useCallback(
    (id: string, index: number, action: () => void) => {
      const resident = getWorkflowLibraryBrowseSnapshot().entries.some((entry) => entry.item.workflow_id === id);

      if (isWindowed && !resident) {
        void loadWorkflowLibraryRange(index, index).then((ready) => {
          if (ready) {
            action();
          }
        });
      } else {
        action();
      }
    },
    [isWindowed]
  );
  const actions = useMemo<WorkflowLibraryListRowActions>(
    () => ({
      menu: (entry, index, anchor) => {
        retain(entry, index);
        ensurePosition(entry.item.workflow_id, index, () => onContextMenu(entry.item.workflow_id, anchor));
      },
      open: (id, index) => ensurePosition(id, index, () => onOpen(id)),
      preview: (id, index) =>
        ensurePosition(id, index, () => {
          onSelect(id);
          onPreview?.(id);
        }),
      select: (id, index) => ensurePosition(id, index, () => onSelect(id)),
    }),
    [ensurePosition, onContextMenu, onOpen, onPreview, onSelect, retain]
  );

  const handleDoubleClick = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      const target = event.target as Element;
      const row = rowOf(target);
      const index = row ? positionOf(row) : -1;
      const entry = entryAt(rows, index);

      if (entry && target.closest('[data-list-primary]')) {
        actions.open(entry.item.workflow_id, index);
      }
    },
    [actions, rows]
  );
  // Pointer-only quick actions never take focus: it moves to their row, so a dialog they open returns there.
  const handleMouseDown = useCallback((event: MouseEvent<HTMLDivElement>) => {
    const target = event.target as Element;

    if (target.closest('[data-row-quick-action]')) {
      event.preventDefault();
      rowOf(target)?.querySelector<HTMLElement>('[data-list-primary]')?.focus({ preventScroll: true });
    }
  }, []);
  const handleFocusCapture = useCallback(
    (event: FocusEvent<HTMLDivElement>) => {
      const row = rowOf(event.target);
      const index = row ? positionOf(row) : -1;
      const entry = entryAt(rows, index);

      if (entry) {
        retain(entry, index);
      }
    },
    [retain, rows]
  );

  const canPreview = onPreview !== undefined;
  const renderItem = ({ entry, index }: WorkflowLibraryListItem, rowProps: ListRowProps) => {
    const id = entry?.item.workflow_id;

    return (
      <WorkflowLibraryListRow
        {...rowProps}
        actions={actions}
        canPreview={canPreview}
        entry={entry}
        index={index}
        isActiveWorkflow={id !== undefined && id === activeWorkflowId}
        isInProject={id !== undefined && linkedTemplates.has(id)}
        menuOpenedBy={
          id !== undefined && openMenuAnchor?.workflowId === id
            ? openMenuAnchor.kind === 'trigger'
              ? 'button'
              : 'row'
            : null
        }
        missingCount={id === undefined ? 0 : (missingCounts.get(id) ?? 0)}
        missingNodePackCount={
          templates && entry?.enrichment.status === 'ready'
            ? getMissingNodePacks(entry.enrichment, templates).length
            : null
        }
        showLastRun={!isWindowed}
        showVisibilityStatus={showVisibilityStatus}
        trailingField={trailingField}
      />
    );
  };

  return (
    <Box
      ref={wrapperRef}
      css={LIST_CSS}
      data-workflow-library-collection=""
      data-workflow-library-list=""
      onDoubleClick={handleDoubleClick}
      onFocusCapture={handleFocusCapture}
      onMouseDown={handleMouseDown}
      onScrollCapture={handleScrollCapture}
    >
      {isActive && isWindowed && range !== null && total > 0 ? (
        <WorkflowLibraryDemand key={demandKey} end={last} ids={ids} start={first} />
      ) : null}
      {hasEntries && error ? (
        <HStack color="fg.subtle" flexShrink={0} fontSize="xs" gap="2" justify="center" role="alert">
          <Text>{error}</Text>
          {isWindowed ? (
            <Button size="sm" variant="ghost" onClick={handleRetry}>
              {t('common.retry')}
            </Button>
          ) : null}
        </HStack>
      ) : null}
      <List
        activeKey={activeKey}
        density="comfortable"
        dividers
        emptyState={
          <Text color="fg.subtle" fontSize="md" py="6" textAlign="center">
            {emptyMessage ?? t('workflowLibrary.empty')}
          </Text>
        }
        errorState={
          <HStack color="fg.subtle" fontSize="md" gap="2" py="6" role="alert">
            <Text>{error}</Text>
            {isWindowed ? (
              <Button loading={isReloading} size="sm" variant="ghost" onClick={handleReload}>
                {t('common.retry')}
              </Button>
            ) : null}
          </HStack>
        }
        initialScrollOffset={initialScrollOffset}
        inset="none"
        label={t('workflowLibrary.title')}
        renderItem={renderItem}
        rows={rows}
        status={listStatus}
        surface="bg.panel"
        onRangeChange={isWindowed ? handleRangeChange : undefined}
        onScrollOffsetPersist={isWindowed ? persistScrollOffset : undefined}
      />
    </Box>
  );
};

interface WorkflowLibraryListRowActions {
  menu: (entry: WorkflowLibraryEntry, index: number, anchor: WorkflowCardMenuAnchor) => void;
  open: (id: string, index: number) => void;
  preview: (id: string, index: number) => void;
  select: (id: string, index: number) => void;
}

interface WorkflowLibraryListRowProps extends ListRowProps {
  actions: WorkflowLibraryListRowActions;
  canPreview: boolean;
  /** Null while the position's metadata is not resident: the row is a busy placeholder with the same controls. */
  entry: WorkflowLibraryEntry | null;
  index: number;
  /** The project's active workflow: badged, and offers no Open since it is already open. */
  isActiveWorkflow: boolean;
  isInProject: boolean;
  /** Which of the row's controls has the actions menu open, if any; that control names the menu for the dialog. */
  menuOpenedBy: 'row' | 'button' | null;
  missingCount: number;
  missingNodePackCount: number | null;
  showLastRun: boolean;
  showVisibilityStatus: boolean;
  trailingField: WorkflowLibraryListTrailingField;
}

const ignoreContextMenu = () => {};
const PLACEHOLDER_LEADING = <Box {...PLACEHOLDER_THUMBNAIL_PROPS} data-workflow-list-placeholder="" />;
const PLACEHOLDER_DESCRIPTION = <Box as="span" {...PLACEHOLDER_LINE_PROPS} />;

/**
 * One component for a position whether or not its page is resident, so the ListItem and its primary button persist
 * when the page arrives. Memoised: range loads, enrichment and scrolling re-render the list, but rarely a given row.
 */
const WorkflowLibraryListRow = memo(function WorkflowLibraryListRow({
  actions,
  canPreview,
  entry,
  index,
  isActiveWorkflow,
  isInProject,
  menuOpenedBy,
  missingCount,
  missingNodePackCount,
  showLastRun,
  showVisibilityStatus,
  trailingField,
  ...rowProps
}: WorkflowLibraryListRowProps) {
  const { t } = useTranslation();
  const item = entry?.item ?? null;
  const id = item?.workflow_id ?? null;
  const dataTime = useDataTimeSnapshot(item);

  // With the date column showing the last run, the facts line leads with the node count rather than repeat it.
  const lastRun =
    showLastRun && trailingField !== 'last_run_at' && item?.last_run_at
      ? formatTimeSince(item.last_run_at, dataTime)
      : '';
  const facts = entry
    ? getWorkflowLibraryEntryFacts({ entry, isInProject, lastRun, missingCount, missingNodePackCount }, t)
    : null;
  const timestamp = item && trailingField ? item[trailingField] : null;
  const menuContentId = id === null ? undefined : getWorkflowLibraryCardMenuContentId(id);
  const isPublic = showVisibilityStatus && item?.is_public === true;

  const handlePress = useCallback(() => id !== null && actions.select(id, index), [actions, id, index]);
  const handleContextMenu = useCallback(
    ({ x, y }: ListContextMenuAnchor) =>
      entry !== null && actions.menu(entry, index, { kind: 'point', workflowId: entry.item.workflow_id, x, y }),
    [actions, entry, index]
  );
  const handleOpen = useCallback(() => id !== null && actions.open(id, index), [actions, id, index]);
  const handlePreview = useCallback(() => id !== null && actions.preview(id, index), [actions, id, index]);
  const handleMore = useCallback(
    () =>
      entry !== null &&
      actions.menu(entry, index, {
        kind: 'trigger',
        triggerId: getWorkflowLibraryCardMenuId(entry.item.workflow_id),
        workflowId: entry.item.workflow_id,
      }),
    [actions, entry, index]
  );

  return (
    <ListItem
      {...rowProps}
      actions={
        // A placeholder keeps the same controls, hidden, so the row's width does not change when its page arrives.
        <HStack aria-hidden={entry === null || undefined} gap="1" visibility={entry === null ? 'hidden' : undefined}>
          <Button
            aria-hidden
            css={ROW_ACTION_CSS}
            data-row-action="open"
            data-row-quick-action
            // The active project workflow is already open; its rail disables Open too.
            data-unavailable={entry === null || isActiveWorkflow || undefined}
            size="xs"
            tabIndex={-1}
            variant="outline"
            onClick={handleOpen}
          >
            {/* Parity with the rail: opening asks what to do once the project holds a copy. */}
            {isInProject ? t('workflowLibrary.openWithEllipsis') : t('workflowLibrary.open')}
          </Button>
          {canPreview ? (
            <Tooltip content={t('workflowLibrary.preview.actionHint')}>
              <IconButton
                aria-hidden
                aria-label={t('workflowLibrary.preview.action')}
                css={ROW_ACTION_CSS}
                data-row-action="preview"
                data-row-quick-action
                data-unavailable={entry === null || undefined}
                size="xs"
                tabIndex={-1}
                variant="outline"
                onClick={handlePreview}
              >
                <WorkflowIcon />
              </IconButton>
            </Tooltip>
          ) : null}
          {/* The row's keyboard control after its button: it shares the row's roving tab stop. */}
          <Tooltip content={t('workflowLibrary.moreActions')}>
            <IconButton
              aria-controls={menuOpenedBy === 'button' ? menuContentId : undefined}
              aria-expanded={menuOpenedBy === 'button'}
              aria-haspopup="menu"
              aria-label={t('workflowLibrary.moreActions')}
              css={ROW_ACTION_CSS}
              data-workflow-card-menu={id ?? undefined}
              id={id === null ? undefined : getWorkflowLibraryCardMenuId(id)}
              size="xs"
              tabIndex={entry === null ? -1 : rowProps.tabIndex}
              variant="outline"
              onClick={handleMore}
            >
              <EllipsisIcon />
            </IconButton>
          </Tooltip>
        </HStack>
      }
      badges={
        isPublic || isActiveWorkflow || isInProject ? (
          <>
            {isPublic ? (
              <Badge data-workflow-visibility="public" variant="subtle">
                {t('workflowLibrary.public')}
              </Badge>
            ) : null}
            {isActiveWorkflow ? (
              <Badge data-active-workflow variant="solid">
                {t('workflowLibrary.activeWorkflow')}
              </Badge>
            ) : null}
            {isInProject ? (
              <Box as="span" data-workflow-in-project display="inline-flex" opacity={0.8}>
                <Icon aria-hidden as={FolderCheckIcon} boxSize="3" />
                <Text srOnly>{t('workflowLibrary.inProject')}</Text>
              </Box>
            ) : null}
          </>
        ) : null
      }
      controls={menuOpenedBy === 'row' ? menuContentId : undefined}
      description={
        facts ? <WorkflowLibraryFactsLine facts={facts} isOnAccent={rowProps.isActive} /> : PLACEHOLDER_DESCRIPTION
      }
      id={id === null ? undefined : getWorkflowLibraryCardId(id)}
      isBusy={entry === null || rowProps.isBusy}
      isExpanded={menuOpenedBy === 'row' || undefined}
      isMenuOpen={menuOpenedBy !== null}
      leading={
        entry ? (
          // 54px wide, so the 3:2 well is the 36px a comfortable row's media takes.
          <WorkflowSummaryThumbnail
            src={entry.item.thumbnail_url}
            w="3.375rem"
            workflowDocument={entry.enrichment.status === 'ready' ? entry.enrichment.document : null}
          />
        ) : (
          PLACEHOLDER_LEADING
        )
      }
      // A placeholder has no name yet; its button says it is loading.
      pressLabel={entry === null ? t('workflowLibrary.loading') : undefined}
      title={item ? item.name || t('workflowLibrary.untitled') : ''}
      titleTruncate="end"
      trailing={timestamp ? formatTimeSince(timestamp, dataTime) : null}
      // A placeholder has no actions yet; it still keeps the browser's own menu from opening over the list.
      onContextMenu={entry === null ? ignoreContextMenu : handleContextMenu}
      onPress={handlePress}
    />
  );
});
