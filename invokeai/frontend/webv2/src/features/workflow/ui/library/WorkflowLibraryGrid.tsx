import type { WorkflowLibraryBrowseSnapshot, WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';

import { Box, type SystemStyleObject, Text } from '@chakra-ui/react';
import {
  getWorkflowLibraryBrowseSnapshot,
  loadWorkflowLibraryRange,
  rememberWorkflowLibraryScroll,
  setWorkflowLibraryVisibleDemand,
} from '@features/workflow/data/libraryBrowseStore';
import { useInvocationTemplatesSnapshot } from '@features/workflow/react';
import { useWorkflowProjectSelector } from '@features/workflow/ui/WorkflowUiContext';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { Scrollable } from '@platform/ui';
import { Button } from '@platform/ui/Button';
import {
  type ComponentProps,
  type ReactNode,
  use,
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { defaultRangeExtractor, useVirtualizer, type VirtualItem, type Virtualizer } from 'react-hook-tanstack-virtual';
import { useTranslation } from 'react-i18next';

import {
  WorkflowLibraryCard,
  getWorkflowLibraryCardId,
  getWorkflowLibraryCardMenuId,
  type WorkflowCardContextMenuHandler,
  type WorkflowCardMenuAnchor,
} from './WorkflowLibraryCard';
import {
  getWorkflowLibraryCardWidth,
  getWorkflowLibraryGridBounds,
  getWorkflowLibraryGridColumns,
  GRID_GAP_PX,
} from './workflowLibraryGridLayout';
import { getMissingNodePacks } from './WorkflowRequirementsList';
import { WORKFLOW_THUMBNAIL_ASPECT_RATIO, WorkflowLibraryCardSizeContext } from './workflowThumbnail';

/**
 * What the grid renders from: the mounted rows and the visible range, never the scroll offset, so scrolling within a
 * row re-renders nothing and only crossing a row boundary or a measurement does. Scroll start and stop count only
 * while a restoration waits for scrolling to settle. The methods, the element cache and the instance are stable
 * across snapshots.
 */
type GridVirtualizer = Virtualizer<HTMLDivElement, Element>;

const selectVirtualizerView = (
  watchesScrollEnd: boolean,
  instance: GridVirtualizer,
  snapshot: Pick<
    GridVirtualizer,
    | 'cancelScroll'
    | 'elementsCache'
    | 'isScrolling'
    | 'measure'
    | 'measureElement'
    | 'range'
    | 'scrollRect'
    | 'scrollToIndex'
    | 'scrollToOffset'
  > & { totalSize: number; virtualItems: VirtualItem[] }
) => ({
  cancelScroll: snapshot.cancelScroll,
  elementsCache: snapshot.elementsCache,
  instance,
  // Not read by the grid: a change re-renders it, so the restoration effect sees the scroll it started settle.
  isScrolling: watchesScrollEnd && snapshot.isScrolling,
  measure: snapshot.measure,
  measureElement: snapshot.measureElement,
  rangeEnd: snapshot.range?.endIndex ?? 0,
  rangeStart: snapshot.range?.startIndex ?? 0,
  scrollToIndex: snapshot.scrollToIndex,
  scrollToOffset: snapshot.scrollToOffset,
  totalSize: snapshot.totalSize,
  viewportHeight: snapshot.scrollRect?.height ?? 0,
  virtualItems: snapshot.virtualItems,
});

/**
 * A card's height below its image at the default 16px root size: borders, body padding, title and facts rows, and
 * the row's bottom gap come to 78px; the 3:2 image is 2px narrower than the card. Rounding down keeps the estimate
 * a lower bound, so a viewport capped at N estimated rows never spans more than N + 1 real ones.
 */
const ROW_CHROME_PX = 76;
const NO_MISSING_COUNTS: ReadonlyMap<string, number> = new Map();
const NO_LINKED_TEMPLATES: ReadonlySet<string> = new Set();

/** The library templates the project holds copies of, as one string so unrelated project edits keep it equal. */
const selectLinkedTemplateKey = (project: { workflows: readonly { source?: { libraryWorkflowId: string } }[] }) =>
  project.workflows
    .flatMap((workflow) => (workflow.source ? [workflow.source.libraryWorkflowId] : []))
    .sort()
    .join('\n');
// A flow layout still needs a finite virtual viewport, independent of the collection's total row height.
const GRID_CSS = {
  '@container workflow-library-content ((width < 32rem) and (height < 32rem)) or (height < 20rem)': {
    flex: '1 1 16rem',
    height: '16rem',
  },
} as const;
/**
 * The scroll area writes its overflow offsets as inherited custom properties on the viewport at every scroll event,
 * which restyles every element below it. The grid uses none of them; pinning them on the content wrapper stops that
 * restyle at the wrapper, which dominated scrolling (about half the main thread) before.
 */
const CONTENT_PROPS = {
  css: {
    '--scroll-area-overflow-x-start': '0px',
    '--scroll-area-overflow-x-end': '0px',
    '--scroll-area-overflow-y-start': '0px',
    '--scroll-area-overflow-y-end': '0px',
  },
};

export interface WorkflowLibraryGridProps {
  /** The project's active workflow, marked on its card in the This-project view. */
  activeWorkflowId?: string | null;
  entries: readonly WorkflowLibraryEntry[];
  /** What an empty result says; defaults to no matches for the filters. */
  emptyMessage?: string;
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
  onContextMenu: WorkflowCardContextMenuHandler;
  onOpen: (workflowId: string) => void;
  /** Offers a Preview quick action on each card; absent hides it. */
  onPreview?: (workflowId: string) => void;
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
  emptyMessage,
  entries,
  window: browseWindow,
  error,
  missingCounts = NO_MISSING_COUNTS,
  openMenuAnchor = null,
  showVisibilityStatus = false,
  onContextMenu,
  onOpen,
  onPreview,
  onSelect,
  selectedWorkflowId,
  removedWorkflowId = null,
  status,
}: WorkflowLibraryGridProps) => {
  const { t } = useTranslation();
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const cardSize = use(WorkflowLibraryCardSizeContext);
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const templates = templatesSnapshot.status === 'loaded' ? templatesSnapshot.templates : null;
  // Only library cards mark templates the project holds; the project view is that collection.
  const isProjectCollection = browseWindow === undefined;
  const linkedTemplateKey = useWorkflowProjectSelector(selectLinkedTemplateKey);
  const linkedTemplates = useMemo(
    () => (isProjectCollection || !linkedTemplateKey ? NO_LINKED_TEMPLATES : new Set(linkedTemplateKey.split('\n'))),
    [isProjectCollection, linkedTemplateKey]
  );

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
  const remeasurePending = useRef(false);
  // The ref is read when the virtualizer publishes a snapshot; a restoration starting later shows at the next one.
  const selectView = useCallback(
    (snapshot: Parameters<typeof selectVirtualizerView>[2], instance: GridVirtualizer) =>
      selectVirtualizerView(restorePending.current, instance, snapshot),
    []
  );
  /** The card or menu button that last held focus in the grid, to hand focus to its remount. */
  const lastFocused = useRef<{ element: HTMLElement; id: string; menu: boolean } | null>(null);
  const retainFocus = useCallback(() => {
    const last = lastFocused.current;
    const active = document.activeElement;
    if (!last || last.element.isConnected || (active !== null && active !== document.body)) {
      return;
    }
    // A column change moves the card into another row, which mounts it afresh under the same id.
    const next = document.getElementById(
      last.menu ? getWorkflowLibraryCardMenuId(last.id) : getWorkflowLibraryCardId(last.id)
    );
    if (next) {
      lastFocused.current = { ...last, element: next };
      next.focus({ preventScroll: true });
    }
  }, []);
  const mounted = useRef(false);
  const positionedAt = useRef<{ row: number; width: number } | null>(null);
  const [focusCard, setFocusCard] = useState<{ entry: WorkflowLibraryEntry; index: number } | null>(null);
  if (focusCard?.entry.item.workflow_id === removedWorkflowId && focusCard !== null) {
    setFocusCard(null);
  }
  const visibleEntries = useMemo(() => dedupeByWorkflowId(entries), [entries]);
  const startIndex = browseWindow?.startIndex ?? 0;
  const total = browseWindow?.total ?? visibleEntries.length;
  const columns = getWorkflowLibraryGridColumns(width, cardSize);
  const { maxViewportRows, overscanRows } = getWorkflowLibraryGridBounds(columns);
  const rowHeight = getWorkflowLibraryCardWidth(width, columns) / WORKFLOW_THUMBNAIL_ASPECT_RATIO + ROW_CHROME_PX;
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
  const virtualizer = useVirtualizer(
    {
      count: Math.ceil(total / columns),
      estimateSize: () => rowHeight,
      getScrollElement,
      enabled: width > 0,
      onChange: (instance) => {
        if (remeasurePending.current && !instance.isScrolling) {
          remeasurePending.current = false;
          for (const element of instance.elementsCache.values()) {
            instance.measureElement(element);
          }
        }
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
      overscan: overscanRows,
      rangeExtractor,
      useFlushSync: false,
    },
    selectView
  );
  const { measure, measureElement, elementsCache, instance, virtualItems, totalSize, cancelScroll } = virtualizer;
  useMountEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }
    mounted.current = true;
    const observer = new MutationObserver(() => {
      retainFocus();
      attemptFocus();
    });
    observer.observe(viewport, { childList: true, subtree: true });
    // Focus leaving the grid for another element ends retention; a removed card's focus falls to the body instead.
    const releaseFocus = (event: FocusEvent) => {
      if (!(event.target instanceof Node) || !viewport.contains(event.target)) {
        lastFocused.current = null;
      }
    };
    document.addEventListener('focusin', releaseFocus);
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
      document.removeEventListener('focusin', releaseFocus);
    };
  });
  useLayoutEffect(() => {
    measure();
  }, [measure, rowHeight]);
  useLayoutEffect(() => {
    // Clearing sizes does not retrigger ResizeObserver for unchanged mounted rows. measureElement defers reads during
    // scrolling, so the virtualizer's change callback retries once scrolling settles.
    if (instance.isScrolling) {
      remeasurePending.current = true;
      return;
    }
    remeasurePending.current = false;
    for (const element of elementsCache.values()) {
      measureElement(element);
    }
  }, [measureElement, elementsCache, instance, rowHeight]);
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
  const firstVisibleRow = restoreRow ?? virtualizer.rangeStart;
  const lastVisibleRow =
    restoreRow === null ? virtualizer.rangeEnd : restoreRow + Math.ceil(virtualizer.viewportHeight / rowHeight);
  const visibleFirst = firstVisibleRow * columns;
  const visibleLast = Math.min(total - 1, (lastVisibleRow + 1) * columns - 1);
  const first = Math.max(0, firstVisibleRow - overscanRows) * columns;
  const last = Math.min(total - 1, (lastVisibleRow + overscanRows + 1) * columns - 1);
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
  const handlePreview = useCallback(
    (id: string) => ensureCardPosition(id, () => onPreview?.(id)),
    [ensureCardPosition, onPreview]
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
      // Cards carry no handlers (they stay memoised across window and enrichment updates); their events land here.
      onClick: (event) => {
        const target = event.target as Element;
        const action = target.closest<HTMLElement>('[data-card-action]');
        const id = target.closest<HTMLElement>('[data-workflow-card-root]')?.dataset.workflowCardRoot;
        if (!id) {
          return;
        }
        if (action?.dataset.cardAction === 'open') {
          handleOpen(id);
        } else if (action?.dataset.cardAction === 'preview') {
          handleSelect(id);
          handlePreview(id);
        } else if (target.closest('[data-workflow-card-menu]')) {
          handleContextMenu(id, { kind: 'trigger', triggerId: getWorkflowLibraryCardMenuId(id), workflowId: id });
        } else if (target.closest('[data-workflow-card]')) {
          handleSelect(id);
        }
      },
      onDoubleClick: (event) => {
        const id = (event.target as Element).closest<HTMLElement>('[data-workflow-card]')?.dataset.workflowCard;
        if (id) {
          handleOpen(id);
        }
      },
      onContextMenu: (event) => {
        const target = event.target as Element;
        if (!target.closest('[data-workflow-card],[data-workflow-card-actions]')) {
          return;
        }
        const id = target.closest<HTMLElement>('[data-workflow-card-root]')?.dataset.workflowCardRoot;
        if (!id) {
          return;
        }
        event.preventDefault();
        // A keyboard-raised menu (Shift+F10, the Menu key) carries no pointer position; anchor it inside the card
        // instead of at the viewport corner.
        const rect = document.getElementById(getWorkflowLibraryCardId(id))?.getBoundingClientRect();
        handleContextMenu(
          id,
          event.clientX === 0 && event.clientY === 0 && rect
            ? { kind: 'point', workflowId: id, x: rect.left + 16, y: rect.top + 16 }
            : { kind: 'point', workflowId: id, x: event.clientX, y: event.clientY }
        );
      },
      // Pointer-only quick actions never take focus: it moves to their card, so a dialog they open returns there.
      onMouseDown: (event) => {
        const target = event.target as Element;
        const id = target.closest('[data-card-action]')
          ? target.closest<HTMLElement>('[data-workflow-card-root]')?.dataset.workflowCardRoot
          : undefined;
        if (id) {
          event.preventDefault();
          document.getElementById(getWorkflowLibraryCardId(id))?.focus({ preventScroll: true });
        }
      },
      onFocusCapture: (event) => {
        const card = (event.target as HTMLElement).closest<HTMLElement>(
          '[data-workflow-card],[data-workflow-card-menu]'
        );
        const id = card?.dataset.workflowCard ?? card?.dataset.workflowCardMenu;
        if (card && id) {
          lastFocused.current = { element: card, id, menu: card.hasAttribute('data-workflow-card-menu') };
        }
        const index = visibleEntries.findIndex((entry) => entry.item.workflow_id === id);
        const entry = visibleEntries[index];
        if (entry) {
          setFocusCard({ entry: { ...entry, enrichment: { status: 'pending' } }, index: startIndex + index });
        }
      },
    }),
    [
      browseWindow,
      virtualizer,
      visibleEntries,
      startIndex,
      focusCard,
      total,
      columns,
      attemptFocus,
      handleContextMenu,
      handleOpen,
      handlePreview,
      handleSelect,
    ]
  );
  return (
    <Scrollable
      css={GRID_CSS}
      flex="1"
      label={t('workflowLibrary.title')}
      minH="0"
      maxH={browseWindow ? `${rowHeight * maxViewportRows}px` : undefined}
      minW="0"
      viewportRef={viewportRef}
      viewportProps={viewportProps}
      contentProps={CONTENT_PROPS}
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
              <GridRow
                key={row.key}
                columns={columns}
                index={row.index}
                measureRef={virtualizer.measureElement}
                start={row.start}
              >
                {Array.from({ length: columns }, (_, column) => {
                  const index = row.index * columns + column;
                  if (index >= total) {
                    return null;
                  }
                  const entry =
                    visibleEntries[index - startIndex] ?? (focusCard?.index === index ? focusCard.entry : undefined);
                  if (!entry) {
                    return <GridPlaceholder key={index} height={rowHeight - GRID_GAP_PX} />;
                  }
                  return (
                    <WorkflowLibraryCard
                      key={entry.item.workflow_id}
                      entry={entry}
                      isActive={entry.item.workflow_id === activeWorkflowId}
                      isInProject={linkedTemplates.has(entry.item.workflow_id)}
                      isSelected={entry.item.workflow_id === selectedWorkflowId}
                      menuOpenedBy={
                        openMenuAnchor?.workflowId === entry.item.workflow_id
                          ? openMenuAnchor.kind === 'trigger'
                            ? 'button'
                            : 'card'
                          : null
                      }
                      missingCount={missingCounts.get(entry.item.workflow_id) ?? 0}
                      missingNodePackCount={
                        templates && entry.enrichment.status === 'ready'
                          ? getMissingNodePacks(entry.enrichment, templates).length
                          : null
                      }
                      showLastRun={isProjectCollection}
                      showVisibilityStatus={showVisibilityStatus}
                      canPreview={onPreview !== undefined}
                    />
                  );
                })}
              </GridRow>
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
            {error ?? emptyMessage ?? t('workflowLibrary.empty')}
          </Text>
        ) : null}
      </Box>
    </Scrollable>
  );
};

const GRID_ROW_CSS: SystemStyleObject = {
  display: 'grid',
  gap: '3',
  left: '0',
  pb: '3',
  position: 'absolute',
  top: '0',
  w: 'full',
};

/**
 * A virtual row. Its offset and column count go in the style attribute: as style props, every new offset would add a
 * CSS rule while scrolling, and each insertion restyles the whole document.
 */
const GridRow = ({
  children,
  columns,
  index,
  measureRef,
  start,
}: {
  children: ReactNode;
  columns: number;
  index: number;
  measureRef: (element: HTMLElement | null) => void;
  start: number;
}) => {
  const style = useMemo(
    () => ({ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`, transform: `translateY(${start}px)` }),
    [columns, start]
  );

  return (
    <Box ref={measureRef} css={GRID_ROW_CSS} data-index={index} style={style}>
      {children}
    </Box>
  );
};

/** A position whose metadata is not resident yet, as tall as a card. */
const GridPlaceholder = ({ height }: { height: number }) => {
  const style = useMemo(() => ({ height: `${height}px` }), [height]);

  return <Box aria-busy="true" bg="bg.subtle" rounded="lg" style={style} />;
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
