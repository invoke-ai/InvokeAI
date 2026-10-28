import type { FocusEvent, ReactNode } from 'react';

import { Box, Flex, HStack, Icon, Text, type SystemStyleObject } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { ArrowLeftIcon, PlusIcon } from 'lucide-react';
import { createContext, useCallback, useContext, useLayoutEffect, useMemo, useRef } from 'react';

import { Button, IconButton } from './Button';
import { Tooltip } from './Tooltip';

/**
 * The list/detail shell shared by the Launchpad managers (models, nodes, fonts, preferences). Side by side when
 * both panes fit; below that one pane at a time, the library first, the detail behind a Back control in its header.
 * A footer (install queue, activity) stays visible under whichever pane shows. The column and the detail header
 * share one header height so their bottom borders line up.
 */

/** The library takes 39% of the manager, between 20rem and 28rem. */
const LIBRARY_WIDTH = 'clamp(20rem, 39cqi, 28rem)';
/**
 * Side by side while the detail keeps 30rem beside a 20rem library (50rem, 800px): the managers' detail headers
 * wrap their actions under the title and shorten the item tab below that width. Measured on the manager's own box,
 * so it also answers when the Launchpad rail takes width.
 */
const SINGLE_PANE = '@container manager (width < 50rem)';
const SINGLE_PANE_FLAG = '--manager-single-pane';
const HEADER_MIN_HEIGHT = '2.75rem';
/** Persistent popups whose closing hands focus back to a trigger the pane switch may have hidden. */
const POPUP_SELECTOR = '[role="dialog"], [role="alertdialog"], [role="menu"], [role="listbox"]';
const TABBABLE_SELECTOR = [
  'button:not(:disabled)',
  'input:not(:disabled)',
  'select:not(:disabled)',
  'textarea:not(:disabled)',
  'a[href]',
  '[tabindex="0"]',
]
  .map((selector) => `${selector}:not([tabindex="-1"])`)
  .join(', ');
/** How long a pane switch made from a popup waits for that popup to return focus. */
const HANDOFF_TIMEOUT_MS = 4000;
/**
 * How long focus is watched after it moves with a pane: the list re-keys its virtual rows a frame or two after its
 * data changes, which can unmount the row just focused.
 */
const SETTLE_MS = 300;

const ROOT_CSS: SystemStyleObject = {
  containerName: 'manager',
  containerType: 'inline-size',
  h: 'full',
  minH: '0',
  w: 'full',
};

/**
 * Both panes stay mounted. In single-pane mode they share one grid cell and the other is visibility-hidden: out of
 * the accessibility tree and tab order, but still laid out, so scroll positions, virtualized rows and form state
 * survive and nothing remounts or refetches. Side by side the footer sits under the detail; in single-pane mode it
 * spans the manager under both.
 */
const GRID_CSS: SystemStyleObject = {
  display: 'grid',
  gridTemplateColumns: `${LIBRARY_WIDTH} minmax(0, 1fr)`,
  gridTemplateRows: 'minmax(0, 1fr) auto',
  h: 'full',
  [SINGLE_PANE_FLAG]: '0',
  '& > [data-manager-pane]': { display: 'flex', flexDirection: 'column', minH: '0', minW: '0', overflow: 'hidden' },
  '& > [data-manager-pane="library"]': { borderEndWidth: '1px', gridColumn: '1', gridRow: '1 / -1' },
  '& > [data-manager-pane="detail"]': { gridColumn: '2', gridRow: '1' },
  '& > [data-manager-footer]': { display: 'flex', flexDirection: 'column', gridColumn: '2', gridRow: '2', minH: '0' },
  '& [data-manager-single-pane-only]': { display: 'none' },
  '&[data-footer-fills]': { gridTemplateRows: '0 minmax(0, 1fr)' },
  '&[data-footer-fills] > [data-manager-pane="detail"]': { visibility: 'hidden' },
  [SINGLE_PANE]: {
    gridTemplateColumns: 'minmax(0, 1fr)',
    [SINGLE_PANE_FLAG]: '1',
    '& > [data-manager-pane]': { gridColumn: '1', gridRow: '1' },
    '& > [data-manager-pane="library"]': { borderEndWidth: '0' },
    '& > [data-manager-footer]': { gridColumn: '1' },
    '& [data-manager-single-pane-only]': { display: 'inline-flex' },
    '&[data-view="library"] > [data-manager-pane="detail"], &[data-view="detail"] > [data-manager-pane="library"], &[data-footer-fills] > [data-manager-pane]':
      { visibility: 'hidden' },
  },
};

interface ManagerBack {
  label: string;
  onBack: () => void;
}

const BACK_SELECTOR = '[data-manager-back]';

/** Lets the detail header place the layout's Back control at its start. */
const ManagerBackContext = createContext<ManagerBack | null>(null);

const isShown = (element: Element): boolean => element.checkVisibility({ visibilityProperty: true });

/**
 * Where Back lands: the open row, else a non-row control last used in the library (the Add shortcut, search), else
 * the list's roving row, else the first control. Rows are found by the list's own state, never by a remembered
 * element: virtual rows are re-keyed and recycled.
 */
const findLibraryEntry = (library: HTMLElement, lastFocused: HTMLElement | null): HTMLElement | null =>
  library.querySelector<HTMLElement>('[data-list-primary][aria-current="true"]') ??
  (lastFocused?.isConnected &&
  library.contains(lastFocused) &&
  !lastFocused.closest('[data-list-row]') &&
  isShown(lastFocused)
    ? lastFocused
    : null) ??
  library.querySelector<HTMLElement>('[data-list-primary][tabindex="0"]') ??
  [...library.querySelectorAll<HTMLElement>(TABBABLE_SELECTOR)].find(isShown) ??
  null;

/** Back while it shows, else the selected tab, else the detail's first control. */
const findDetailEntry = (detail: HTMLElement): HTMLElement | null =>
  [...detail.querySelectorAll<HTMLElement>(BACK_SELECTOR)].find(isShown) ??
  [...detail.querySelectorAll<HTMLElement>('[role="tab"][aria-selected="true"]')].find(isShown) ??
  [...detail.querySelectorAll<HTMLElement>(TABBABLE_SELECTOR)].find(isShown) ??
  null;

export interface ManagerLayoutProps {
  /** The library pane, usually a {@link ManagerColumn}. */
  library: ReactNode;
  /** The detail pane, headed by a {@link ManagerDetailHeader}, which carries Back in single-pane mode. */
  detail: ReactNode;
  /** Progress that must stay in view whichever pane shows: an install queue, recent activity, upload results. */
  footer?: ReactNode;
  /** The footer takes the panes' place, as a maximized queue does. */
  footerFills?: boolean;
  /**
   * Whether the detail is the pane shown when only one fits: set it when the user opens an item or a detail tab,
   * clear it in `onBack`. Side by side it changes nothing.
   */
  isDetailOpen: boolean;
  /** Names the library Back returns to, e.g. "Back to models". */
  backLabel: string;
  onBack: () => void;
}

/**
 * Focus follows the pane switch in single-pane mode: opening moves it to Back, Back returns it to the row it came
 * from. It never stays on a hidden element: a resize across the threshold, or a dialog or menu returning focus to a
 * trigger the switch hid, hands it to the shown pane's entry instead.
 */
export const ManagerLayout = ({
  backLabel,
  detail,
  footer,
  footerFills = false,
  isDetailOpen,
  library,
  onBack,
}: ManagerLayoutProps) => {
  const gridRef = useRef<HTMLDivElement>(null);
  const libraryRef = useRef<HTMLDivElement>(null);
  const detailRef = useRef<HTMLDivElement>(null);
  const lastFocus = useRef<HTMLElement | null>(null);
  const lastLibraryFocus = useRef<HTMLElement | null>(null);
  const shownView = useRef(isDetailOpen);
  const isSinglePane = useRef<boolean | null>(null);
  const handoff = useRef({ deadline: 0, frame: 0, settleUntil: 0 });
  const back = useMemo<ManagerBack>(() => ({ label: backLabel, onBack }), [backLabel, onBack]);

  const readSinglePane = (): boolean =>
    gridRef.current !== null && getComputedStyle(gridRef.current).getPropertyValue(SINGLE_PANE_FLAG).trim() === '1';

  /** The entry of the pane the user should be in: the shown one, or side by side the one focus was last in. */
  const focusEntry = (): boolean => {
    const libraryPane = libraryRef.current;
    const detailPane = detailRef.current;

    if (!libraryPane || !detailPane) {
      return false;
    }

    const toDetail = readSinglePane() ? shownView.current : !libraryPane.contains(lastFocus.current);
    const target = toDetail ? findDetailEntry(detailPane) : findLibraryEntry(libraryPane, lastLibraryFocus.current);

    target?.focus();

    return target !== null && document.activeElement === target;
  };

  /**
   * Moves focus off a hidden element here, or back from the body when it was lost from this manager or, while
   * `isSwitching`, lost to anything at all.
   */
  const repairFocus = (isSwitching: boolean): boolean => {
    const grid = gridRef.current;
    const active = document.activeElement as HTMLElement | null;

    if (!grid) {
      return false;
    }

    const isLost = active === null || active === document.body;
    const isHiddenHere = active !== null && grid.contains(active) && !isShown(active);
    const lostFromHere =
      isLost && lastFocus.current !== null && grid.contains(lastFocus.current) && !isShown(lastFocus.current);

    return (isHiddenHere || (isLost && (isSwitching || lostFromHere))) && focusEntry();
  };

  /**
   * After a switch, watch focus for a moment: a dialog or menu the switch was made from (a confirmed delete, an Open
   * item) still holds focus and will return it to a trigger the switch may have hidden or removed, and a list may
   * re-key the row just focused. Whatever happens, focus ends in the shown pane, never on a hidden element or the
   * body. `popupTimeoutMs` extends the watch while a popup holds focus.
   */
  const watchFocus = (popupTimeoutMs = 0) => {
    const now = performance.now();

    handoff.current.deadline = Math.max(handoff.current.deadline, now + popupTimeoutMs);
    handoff.current.settleUntil = Math.max(handoff.current.settleUntil, now + SETTLE_MS);

    if (handoff.current.frame !== 0) {
      return;
    }

    const tick = () => {
      handoff.current.frame = 0;
      const grid = gridRef.current;

      if (!grid) {
        return;
      }

      const active = document.activeElement as HTMLElement | null;
      const isInPopup = active !== null && !grid.contains(active) && active.closest(POPUP_SELECTOR) !== null;
      const isWaiting = isInPopup && performance.now() < handoff.current.deadline;

      if (!isWaiting) {
        const shownPane = shownView.current ? detailRef.current : libraryRef.current;
        // Returned to a control outside the shown pane (the footer, a hidden trigger) or lost: lead into the pane.
        const isOutsideShownPane =
          active !== null && grid.contains(active) && shownPane !== null && !shownPane.contains(active);

        if (repairFocus(true) || (isOutsideShownPane && readSinglePane() && focusEntry())) {
          handoff.current.settleUntil = performance.now() + SETTLE_MS;
        }
      }

      if (isWaiting || performance.now() < handoff.current.settleUntil) {
        handoff.current.frame = requestAnimationFrame(tick);
      }
    };

    handoff.current.frame = requestAnimationFrame(tick);
  };

  // Focus moves with the pane before paint, so the ring never sits on a pane that just went away; reading the
  // switched panes' computed visibility is the pre-paint measurement.
  useLayoutEffect(() => {
    if (shownView.current === isDetailOpen) {
      return;
    }

    shownView.current = isDetailOpen;
    const grid = gridRef.current;

    if (!grid || !readSinglePane()) {
      return;
    }

    const active = document.activeElement;
    const isLost = active === null || active === document.body;

    // A switch nobody here asked for (the starting pane settling on load) leaves an untouched page alone.
    if (isLost && lastFocus.current === null) {
      return;
    }

    if (isLost || grid.contains(active)) {
      // From the pane being hidden, or the footer, which stays in view but still leads into the shown pane.
      if (!(isDetailOpen ? detailRef.current : libraryRef.current)?.contains(active)) {
        focusEntry();
      }

      watchFocus();
    } else if (active.closest(POPUP_SELECTOR)) {
      watchFocus(HANDOFF_TIMEOUT_MS);
    }
  }, [isDetailOpen]);

  // A filling footer hides the panes; focus left in one (a maximize requested from elsewhere) moves to the footer.
  useLayoutEffect(() => {
    const grid = gridRef.current;
    const active = document.activeElement;

    if (!footerFills || !grid || !active || !grid.contains(active) || isShown(active)) {
      return;
    }

    const footerElement = grid.querySelector<HTMLElement>('[data-manager-footer]');
    [...(footerElement?.querySelectorAll<HTMLElement>(TABBABLE_SELECTOR) ?? [])].find(isShown)?.focus();
  }, [footerFills]);

  // Crossing the threshold (a resize, browser zoom) hides one pane or the Back control without a render.
  useMountEffect(() => {
    const grid = gridRef.current;

    if (!grid) {
      return undefined;
    }

    isSinglePane.current = readSinglePane();
    const observer = new ResizeObserver(() => {
      const next = readSinglePane();

      if (next !== isSinglePane.current) {
        isSinglePane.current = next;
        repairFocus(false);
        watchFocus();
      }
    });

    observer.observe(grid);

    return () => {
      observer.disconnect();
      cancelAnimationFrame(handoff.current.frame);
    };
  });

  const handleFocus = useCallback((event: FocusEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement;

    lastFocus.current = target;

    if (libraryRef.current?.contains(target)) {
      lastLibraryFocus.current = target;
    }
  }, []);

  return (
    <ManagerBackContext.Provider value={back}>
      <Box css={ROOT_CSS}>
        <Box
          ref={gridRef}
          css={GRID_CSS}
          data-footer-fills={footerFills ? '' : undefined}
          data-view={isDetailOpen ? 'detail' : 'library'}
          onFocusCapture={handleFocus}
        >
          <Box ref={libraryRef} data-manager-pane="library">
            {library}
          </Box>
          <Box ref={detailRef} data-manager-pane="detail">
            {detail}
          </Box>
          {footer ? <Box data-manager-footer="">{footer}</Box> : null}
        </Box>
      </Box>
    </ManagerBackContext.Provider>
  );
};

export interface ManagerAddAction {
  /** The Add tab's label, e.g. "Add Models". */
  label: string;
  onAdd: () => void;
}

export const ManagerColumn = ({
  actions,
  addAction,
  children,
  count,
  title,
}: {
  /** Controls at the end of the column header. */
  actions?: ReactNode;
  /**
   * Opens the detail on its Add tab. Shown only in single-pane mode, where the detail's tabs are out of reach from
   * the library; side by side the tab itself is the entry.
   */
  addAction?: ManagerAddAction;
  children: ReactNode;
  count?: ReactNode;
  title: string;
}) => (
  <Flex direction="column" flex="1" minH="0" position="relative">
    <HStack
      borderBottomWidth="1px"
      columnGap="2"
      flexShrink={0}
      flexWrap="wrap"
      minH={HEADER_MIN_HEIGHT}
      px="3"
      py="1"
      rowGap="1"
    >
      <HStack flex="1 1 auto" gap="2" minW="0">
        <Text as="h2" fontSize="lg" fontWeight="700" minW="0" truncate>
          {title}
        </Text>
        {count === undefined ? null : (
          <Text color="fg.muted" flexShrink={0} fontVariantNumeric="tabular-nums">
            {count}
          </Text>
        )}
      </HStack>
      {addAction || actions ? (
        <HStack flexShrink={0} gap="1" ms="auto">
          {addAction ? (
            <Button data-manager-single-pane-only="" size="sm" variant="ghost" onClick={addAction.onAdd}>
              <Icon as={PlusIcon} boxSize="3" />
              {addAction.label}
            </Button>
          ) : null}
          {actions}
        </HStack>
      ) : null}
    </HStack>
    {children}
  </Flex>
);

/** The tabs never wrap a label: the item tab shrinks (it carries a MiddleTruncate), the rest keep their width. */
const DETAIL_HEADER_CSS: SystemStyleObject = {
  '& [role="tablist"]': { flexWrap: 'nowrap', minW: '0' },
  '& [role="tab"]': { flexShrink: 0, whiteSpace: 'nowrap' },
  '& [role="tab"][data-manager-item-tab]': { flexShrink: 1, minW: '0' },
};

/**
 * Holds the detail pane's tabs or title, bottom-aligned so a tab list's indicator sits on the border. In
 * single-pane mode it starts with the layout's Back control. Mark the item tab `data-manager-item-tab` so it is the
 * one that shrinks.
 */
export const ManagerDetailHeader = ({ children }: { children: ReactNode }) => {
  const back = useContext(ManagerBackContext);

  return (
    <Flex
      align="flex-end"
      borderBottomWidth="1px"
      css={DETAIL_HEADER_CSS}
      flexShrink={0}
      gap="1"
      minH={HEADER_MIN_HEIGHT}
      minW="0"
      px="2"
    >
      {back ? (
        <Tooltip content={back.label}>
          <IconButton
            alignSelf="center"
            aria-label={back.label}
            data-manager-back=""
            data-manager-single-pane-only=""
            flexShrink={0}
            size="sm"
            variant="ghost"
            onClick={back.onBack}
          >
            <Icon as={ArrowLeftIcon} boxSize="3.5" />
          </IconButton>
        </Tooltip>
      ) : null}
      {children}
    </Flex>
  );
};
