import type { WorkflowLibraryBrowseSnapshot, WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';

import { Box, Text } from '@chakra-ui/react';
import {
  getWorkflowLibraryBrowseSnapshot,
  loadWorkflowLibraryRange,
  rememberWorkflowLibraryScroll,
  setWorkflowLibraryVisibleDemand,
} from '@features/workflow/data/libraryBrowseStore';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { Scrollable } from '@platform/ui';
import { Button } from '@platform/ui/Button';
import { type ComponentProps, useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { defaultRangeExtractor, useVirtualizer } from 'react-hook-tanstack-virtual';
import { useTranslation } from 'react-i18next';

import {
  WorkflowLibraryCard,
  getWorkflowLibraryCardId,
  getWorkflowLibraryCardMenuId,
  type WorkflowCardMenuAnchor,
  type WorkflowLibraryCardProps,
} from './WorkflowLibraryCard';

/** Two rows before and after the viewport, plus one focused/menu row. */
const OVERSCAN_ROWS = 2;
// A partial row and four overscan rows still fit 60 metadata positions at three columns.
const MAX_VISIBLE_ROWS = 15;
const NO_MISSING_COUNTS: ReadonlyMap<string, number> = new Map();
// A flow layout still needs a finite virtual viewport, independent of the collection's total row height.
const GRID_CSS = {
  '@container workflow-library-content ((width < 32rem) and (height < 32rem)) or (height < 20rem)': {
    flex: '1 1 16rem',
    height: '16rem',
  },
} as const;

export interface WorkflowLibraryGridProps {
  /** The project's active workflow, marked on its card in the This-project view. */
  activeWorkflowId?: string | null;
  entries: readonly WorkflowLibraryEntry[];
  window?: {
    startIndex: number;
    total: number;
    filter: WorkflowLibraryBrowseSnapshot['filter'];
  };
  error: string | null;
  /** What has the actions menu open, so that tile's control can name it. */
  openMenuAnchor?: WorkflowCardMenuAnchor | null;
  /** Project workflows reuse the grid but have no library visibility state. */
  showVisibilityStatus?: boolean;
  /** Missing-model counts by workflow id; absent ids render no badge. */
  missingCounts?: ReadonlyMap<string, number>;
  selectedWorkflowId: string | null;
  removedWorkflowId?: string | null;
  status: WorkflowLibraryBrowseSnapshot['status'];
  onContextMenu: WorkflowLibraryCardProps['onContextMenu'];
  onOpen: (workflowId: string) => void;
  onSelect: (workflowId: string) => void;
}

/** Keep duplicate server identities from corrupting card reconciliation. */
const dedupeByWorkflowId = (entries: readonly WorkflowLibraryEntry[]): WorkflowLibraryEntry[] => {
  const seen = new Set<string>();
  const unique: WorkflowLibraryEntry[] = [];

  for (const entry of entries) {
    if (!seen.has(entry.item.workflow_id)) {
      seen.add(entry.item.workflow_id);
      unique.push(entry);
    }
  }

  return unique;
};

export const WorkflowLibraryGrid = ({
  activeWorkflowId = null,
  entries,
  window: browseWindow,
  error,
  missingCounts = NO_MISSING_COUNTS,
  openMenuAnchor = null,
  showVisibilityStatus = false,
  onContextMenu,
  onOpen,
  onSelect,
  selectedWorkflowId,
  removedWorkflowId = null,
  status,
}: WorkflowLibraryGridProps) => {
  const { t } = useTranslation();
  const viewportRef = useRef<HTMLDivElement | null>(null);

  const pendingFocus = useRef<{
    index: number;
    menu: boolean;
    source: HTMLElement;
    filter: WorkflowLibraryBrowseSnapshot['filter'];
    owner: ReturnType<typeof captureAccountScope>;
  } | null>(null);
  const attemptFocus = useCallback(() => {
    const intent = pendingFocus.current;
    if (!intent) {
      return;
    }
    const snapshot = getWorkflowLibraryBrowseSnapshot();
    if (
      !isAccountScopeCurrent(intent.owner) ||
      snapshot.filter !== intent.filter ||
      document.activeElement !== intent.source
    ) {
      pendingFocus.current = null;
      return;
    }
    const entry = snapshot.entries[intent.index - (snapshot.startIndex ?? 0)];
    if (!entry) {
      return;
    }
    const element = document.getElementById(
      intent.menu
        ? getWorkflowLibraryCardMenuId(entry.item.workflow_id)
        : getWorkflowLibraryCardId(entry.item.workflow_id)
    );
    if (element) {
      pendingFocus.current = null;
      element.focus();
      element.scrollIntoView({ block: 'nearest' });
    }
  }, []);
  const [width, setWidth] = useState(0);
  const [owner] = useState(captureAccountScope);
  const [restoreAnchor, setRestoreAnchor] = useState(() => {
    const snapshot = getWorkflowLibraryBrowseSnapshot();
    return browseWindow?.filter === snapshot.filter ? (snapshot.scrollAnchor ?? null) : null;
  });
  const restorePending = useRef(restoreAnchor !== null);
  const mounted = useRef(false);
  const positionedAt = useRef<{ row: number; width: number } | null>(null);
  const [focusCard, setFocusCard] = useState<{ entry: WorkflowLibraryEntry; index: number } | null>(null);
  if (focusCard?.entry.item.workflow_id === removedWorkflowId && focusCard !== null) {
    setFocusCard(null);
  }
  const visibleEntries = useMemo(() => dedupeByWorkflowId(entries), [entries]);
  const startIndex = browseWindow?.startIndex ?? 0;
  const total = browseWindow?.total ?? visibleEntries.length;
  const columns = Math.max(1, Math.min(3, Math.floor(width / 180)));
  const rowHeight = (Math.max(1, (width - 12 * columns) / columns) * 9) / 16 + 82;
  const getScrollElement = useCallback(() => viewportRef.current, []);
  const rangeExtractor = useCallback(
    (range: Parameters<typeof defaultRangeExtractor>[0]) => {
      const indexes = defaultRangeExtractor(range);
      for (const id of [focusCard?.entry.item.workflow_id, openMenuAnchor?.workflowId]) {
        const index = visibleEntries.findIndex((entry) => entry.item.workflow_id === id);
        if (index >= 0) {
          indexes.push(Math.floor((startIndex + index) / columns));
        } else if (
          focusCard &&
          focusCard.entry.item.workflow_id === id &&
          focusCard.index < total &&
          (focusCard.index < startIndex || focusCard.index >= startIndex + visibleEntries.length)
        ) {
          indexes.push(Math.floor(focusCard.index / columns));
        }
      }
      return [...new Set(indexes)].sort((a, b) => a - b);
    },
    [focusCard, openMenuAnchor, startIndex, visibleEntries, columns, total]
  );
  const virtualizer = useVirtualizer({
    count: Math.ceil(total / columns),
    estimateSize: () => rowHeight,
    getScrollElement,
    enabled: width > 0,
    onChange: (instance) => {
      if (!mounted.current || !browseWindow || !width || restorePending.current || !isAccountScopeCurrent(owner)) {
        return;
      }
      const offset = instance.scrollOffset ?? 0;
      const row = instance.getVirtualItems().find((item) => item.end > offset);
      if (row && row.size > 0) {
        rememberWorkflowLibraryScroll(browseWindow.filter, {
          index: row.index * columns,
          fraction: Math.max(0, Math.min(1, (offset - row.start) / row.size)),
        });
      }
    },
    overscan: OVERSCAN_ROWS,
    rangeExtractor,
    useFlushSync: false,
  });
  const { measure, measureElement, elementsCache, isScrolling, virtualItems, totalSize, cancelScroll } = virtualizer;
  useMountEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }
    mounted.current = true;
    const observer = new MutationObserver(attemptFocus);
    observer.observe(viewport, { childList: true, subtree: true });
    const cancel = () => {
      pendingFocus.current = null;
      if (restorePending.current) {
        restorePending.current = false;
        cancelScroll();
        setRestoreAnchor(null);
      }
    };
    owner.signal.addEventListener('abort', cancel, { once: true });
    document.addEventListener('keydown', cancel, true);
    document.addEventListener('pointerdown', cancel, true);
    document.addEventListener('wheel', cancel, { capture: true, passive: true });
    document.addEventListener('focusin', attemptFocus);
    return () => {
      pendingFocus.current = null;
      // Mount replay keeps uncanceled intent, but must restart the canceled native command.
      mounted.current = false;
      positionedAt.current = null;
      cancelScroll();
      observer.disconnect();
      owner.signal.removeEventListener('abort', cancel);
      document.removeEventListener('keydown', cancel, true);
      document.removeEventListener('pointerdown', cancel, true);
      document.removeEventListener('wheel', cancel, true);
      document.removeEventListener('focusin', attemptFocus);
    };
  });
  useLayoutEffect(() => {
    measure();
  }, [measure, rowHeight]);
  useLayoutEffect(() => {
    // Clearing sizes does not retrigger ResizeObserver for unchanged mounted rows.
    // measureElement defers reads during scrolling, so retry once scrolling settles.
    if (isScrolling) {
      return;
    }
    for (const element of elementsCache.values()) {
      measureElement(element);
    }
  }, [measureElement, elementsCache, rowHeight, isScrolling]);
  useMountEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }
    const observer = new ResizeObserver(() => setWidth(viewport.clientWidth));
    observer.observe(viewport);
    return () => observer.disconnect();
  });
  const restoreRow = restoreAnchor ? Math.floor(Math.min(restoreAnchor.index, Math.max(0, total - 1)) / columns) : null;
  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    if (!mounted.current || !restorePending.current || restoreRow === null || !restoreAnchor || !width || !viewport) {
      return;
    }
    if (!isAccountScopeCurrent(owner)) {
      restorePending.current = false;
      cancelScroll();
      return;
    }
    const firstItem = restoreRow * columns;
    if (
      totalSize === 0 ||
      firstItem < startIndex ||
      Math.min(total, firstItem + columns) > startIndex + visibleEntries.length
    ) {
      return;
    }
    // Wait for resident metadata and the committed sizer before starting an index command.
    // A command against an empty sizer can otherwise outlive cancellation through its pending clamp retry.
    if (positionedAt.current?.row !== restoreRow || positionedAt.current.width !== width) {
      positionedAt.current = { row: restoreRow, width };
      virtualizer.scrollToIndex(restoreRow, { align: 'start' });
      return;
    }
    const row = virtualItems.find((item) => item.index === restoreRow);
    const element = row ? elementsCache.get(row.key) : undefined;
    if (!element) {
      return;
    }
    // Translate the logical anchor using the rendered row, including its actual responsive height and gap.
    const rect = element.getBoundingClientRect();
    if (rect.height === 0) {
      return;
    }
    const delta = rect.top - viewport.getBoundingClientRect().top + restoreAnchor.fraction * rect.height;
    const target = Math.max(0, Math.min(viewport.scrollTop + delta, viewport.scrollHeight - viewport.clientHeight));
    if (Math.abs(target - viewport.scrollTop) > 1) {
      virtualizer.scrollToOffset(target);
      return;
    }
    restorePending.current = false;
    setRestoreAnchor(null);
    const visible = virtualItems.find((item) => item.end > viewport.scrollTop);
    if (browseWindow && visible && visible.size > 0) {
      rememberWorkflowLibraryScroll(browseWindow.filter, {
        index: visible.index * columns,
        fraction: Math.max(0, Math.min(1, (viewport.scrollTop - visible.start) / visible.size)),
      });
    }
  }, [
    browseWindow,
    restoreRow,
    restoreAnchor,
    width,
    owner,
    cancelScroll,
    columns,
    virtualizer,
    virtualItems,
    elementsCache,
    startIndex,
    total,
    totalSize,
    visibleEntries,
  ]);
  // Do not let the mount's unknown width/top range replace the cached window needed for restoration.
  const firstVisibleRow = restoreRow ?? virtualizer.range?.startIndex ?? 0;
  const lastVisibleRow =
    restoreRow === null
      ? (virtualizer.range?.endIndex ?? 0)
      : restoreRow + Math.ceil((virtualizer.scrollRect?.height ?? 0) / rowHeight);
  const visibleFirst = firstVisibleRow * columns;
  const visibleLast = Math.min(total - 1, (lastVisibleRow + 1) * columns - 1);
  const first = Math.max(0, firstVisibleRow - OVERSCAN_ROWS) * columns;
  const last = Math.min(total - 1, (lastVisibleRow + OVERSCAN_ROWS + 1) * columns - 1);
  const ids = useMemo(() => {
    const visible: string[] = [];
    const overscan: string[] = [];
    for (const [index, entry] of visibleEntries.entries()) {
      const position = startIndex + index;
      if (position >= first && position <= last) {
        // The owner's bounded enrichment slots must reach every visible card before either overscan edge.
        const demand = position >= visibleFirst && position <= visibleLast ? visible : overscan;
        demand.push(entry.item.workflow_id);
      }
    }
    return visible.concat(overscan);
  }, [visibleEntries, startIndex, first, last, visibleFirst, visibleLast]);
  const hasEntries = visibleEntries.length > 0;
  // Idle precedes the first open-time load; render loading rather than an empty result.
  const isPending = status === 'idle' || status === 'loading';

  const viewportProps = useMemo<ComponentProps<typeof Scrollable>['viewportProps']>(
    () => ({
      onKeyDown: (event) => {
        if (event.key !== 'Tab' || event.altKey || event.ctrlKey || event.metaKey) {
          return;
        }
        const target = event.target as HTMLElement;
        const isCard = target.hasAttribute('data-workflow-card');
        const isMenu = target.hasAttribute('data-workflow-card-menu');
        if ((!isCard || !event.shiftKey) && (!isMenu || event.shiftKey)) {
          return;
        }
        const id = target.dataset.workflowCard ?? target.dataset.workflowCardMenu;
        const residentIndex = visibleEntries.findIndex((entry) => entry.item.workflow_id === id);
        const index = residentIndex >= 0 ? startIndex + residentIndex : (focusCard?.index ?? -1);
        const next = index + (event.shiftKey ? -1 : 1);
        if (index < 0 || next < 0 || next >= total) {
          return;
        }
        event.preventDefault();
        virtualizer.scrollToIndex(Math.floor(next / columns), { align: 'auto' });
        if (browseWindow) {
          pendingFocus.current = {
            index: next,
            menu: event.shiftKey,
            source: target,
            filter: getWorkflowLibraryBrowseSnapshot().filter,
            owner: captureAccountScope(),
          };
          void loadWorkflowLibraryRange(next, next).then((ready) => {
            if (ready) {
              requestAnimationFrame(attemptFocus);
            }
          });
        } else {
          requestAnimationFrame(() => {
            const entry = visibleEntries[next];
            if (!entry || document.activeElement !== target) {
              return;
            }
            document
              .getElementById(
                event.shiftKey
                  ? getWorkflowLibraryCardMenuId(entry.item.workflow_id)
                  : getWorkflowLibraryCardId(entry.item.workflow_id)
              )
              ?.focus();
          });
        }
      },
      onFocusCapture: (event) => {
        const card = (event.target as HTMLElement).closest<HTMLElement>(
          '[data-workflow-card],[data-workflow-card-menu]'
        );
        const id = card?.dataset.workflowCard ?? card?.dataset.workflowCardMenu;
        const index = visibleEntries.findIndex((entry) => entry.item.workflow_id === id);
        const entry = visibleEntries[index];
        if (entry) {
          setFocusCard({ entry: { ...entry, enrichment: { status: 'pending' } }, index: startIndex + index });
        }
      },
    }),
    [browseWindow, virtualizer, visibleEntries, startIndex, focusCard, total, columns, attemptFocus]
  );
  const handleRetry = useCallback(() => {
    void loadWorkflowLibraryRange(first, last, true);
  }, [first, last]);
  const ensureCardPosition = useCallback(
    (id: string, action: () => void) => {
      const resident = visibleEntries.findIndex((entry) => entry.item.workflow_id === id);
      const index = resident >= 0 ? startIndex + resident : focusCard?.index;
      if (browseWindow && index !== undefined && resident < 0) {
        void loadWorkflowLibraryRange(index, index).then((ready) => {
          if (ready) {
            action();
          }
        });
      } else {
        action();
      }
    },
    [browseWindow, visibleEntries, startIndex, focusCard]
  );
  const handleSelect = useCallback(
    (id: string) => ensureCardPosition(id, () => onSelect(id)),
    [ensureCardPosition, onSelect]
  );
  const handleOpen = useCallback(
    (id: string) => ensureCardPosition(id, () => onOpen(id)),
    [ensureCardPosition, onOpen]
  );
  const handleContextMenu = useCallback(
    (id: string, anchor: WorkflowCardMenuAnchor) => {
      const index = visibleEntries.findIndex((entry) => entry.item.workflow_id === id);
      const entry = visibleEntries[index];
      if (entry) {
        setFocusCard({ entry: { ...entry, enrichment: { status: 'pending' } }, index: startIndex + index });
      }
      ensureCardPosition(id, () => onContextMenu(id, anchor));
    },
    [visibleEntries, startIndex, ensureCardPosition, onContextMenu]
  );
  return (
    <Scrollable
      css={GRID_CSS}
      flex="1"
      label={t('workflowLibrary.title')}
      minH="0"
      maxH={browseWindow ? `${rowHeight * MAX_VISIBLE_ROWS}px` : undefined}
      minW="0"
      viewportRef={viewportRef}
      viewportProps={viewportProps}
    >
      {browseWindow && width > 0 ? (
        <GridDemand key={`${first}:${last}:${ids.join(',')}`} ids={ids} start={first} end={last} />
      ) : null}
      <Box display="flex" flexDirection="column" gap="3" minW="0" pr="2" w="full">
        {hasEntries && error ? (
          // Keep recovery within the viewport even for a failed distant range.
          <Text
            role="alert"
            position="sticky"
            top="0"
            zIndex="1"
            bg="bg.panel"
            color="fg.subtle"
            fontSize="xs"
            py="2"
            textAlign="center"
          >
            {error}
            {browseWindow ? (
              <Button size="sm" variant="ghost" onClick={handleRetry}>
                {t('common.retry')}
              </Button>
            ) : null}
          </Text>
        ) : null}
        {hasEntries ? (
          <Box h={`${totalSize}px`} position="relative" minW="0" w="full">
            {virtualItems.map((row) => (
              <Box
                key={row.key}
                data-index={row.index}
                ref={virtualizer.measureElement}
                display="grid"
                gap="3"
                gridTemplateColumns={`repeat(${columns}, minmax(0, 1fr))`}
                position="absolute"
                top="0"
                left="0"
                w="full"
                pb="3"
                transform={`translateY(${row.start}px)`}
              >
                {Array.from({ length: columns }, (_, column) => {
                  const index = row.index * columns + column;
                  if (index >= total) {
                    return null;
                  }
                  const entry =
                    visibleEntries[index - startIndex] ?? (focusCard?.index === index ? focusCard.entry : undefined);
                  if (!entry) {
                    return <Box key={index} h={`${rowHeight - 12}px`} aria-busy="true" bg="bg.subtle" rounded="lg" />;
                  }
                  return (
                    <WorkflowLibraryCard
                      key={entry.item.workflow_id}
                      entry={entry}
                      isActive={entry.item.workflow_id === activeWorkflowId}
                      isSelected={entry.item.workflow_id === selectedWorkflowId}
                      menuOpenedBy={
                        openMenuAnchor?.workflowId === entry.item.workflow_id
                          ? openMenuAnchor.kind === 'trigger'
                            ? 'button'
                            : 'card'
                          : null
                      }
                      missingCount={missingCounts.get(entry.item.workflow_id) ?? 0}
                      showVisibilityStatus={showVisibilityStatus}
                      onContextMenu={handleContextMenu}
                      onOpen={handleOpen}
                      onSelect={handleSelect}
                    />
                  );
                })}
              </Box>
            ))}
          </Box>
        ) : null}
        {!hasEntries && isPending ? (
          <Text color="fg.subtle" fontSize="md" py="6" textAlign="center">
            {t('workflowLibrary.loading')}
          </Text>
        ) : null}
        {!hasEntries && !isPending ? (
          <Text color="fg.subtle" fontSize="md" py="6" textAlign="center">
            {error ?? t('workflowLibrary.empty')}
          </Text>
        ) : null}
      </Box>
    </Scrollable>
  );
};

// Mount leases keep external-store writes out of render and release detail demand when the viewport leaves.
let demandLease = 0;
const GridDemand = ({ ids, start, end }: { ids: readonly string[]; start: number; end: number }) => {
  useMountEffect(() => {
    const lease = ++demandLease;
    setWorkflowLibraryVisibleDemand(ids);
    void loadWorkflowLibraryRange(start, end);
    return () => {
      queueMicrotask(() => {
        if (lease === demandLease) {
          setWorkflowLibraryVisibleDemand([]);
        }
      });
    };
  });
  return null;
};
