import type { ChangeEvent, FocusEvent, ReactNode } from 'react';

import { Box, Flex, HStack, Icon, Input, InputGroup, Kbd, Menu, Text, type SystemStyleObject } from '@chakra-ui/react';
import { focusSearchOnSlash } from '@platform/browser/searchShortcut';
import { useMountEffect } from '@platform/react/useMountEffect';
import { ArrowLeftIcon, PlusIcon, RotateCcwIcon, SearchIcon, SlidersHorizontalIcon, XIcon } from 'lucide-react';
import { createContext, useCallback, useContext, useLayoutEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { Button, IconButton } from './Button';
import { MenuActionItem } from './Menu';
import { isModalPresent } from './modalPresence';
import { Tooltip, useTooltipTriggerIds } from './Tooltip';

/** List/detail shell for the Launchpad managers: side by side when both panes fit, else one pane with Back. */

const LIBRARY_WIDTH = 'clamp(20rem, 39cqi, 28rem)';
/** Below this the detail headers wrap; measured on the manager's own box, so it also answers to the Launchpad rail. */
const SINGLE_PANE = '@container manager (width < 50rem)';
const SINGLE_PANE_FLAG = '--manager-single-pane';
const HEADER_MIN_HEIGHT = '2.75rem';
const SEARCH_SELECTOR = '[data-manager-search]';
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
const HANDOFF_TIMEOUT_MS = 4000;
/** The list re-keys its virtual rows a frame or two after data changes, which can unmount the row just focused. */
const SETTLE_MS = 300;

const ROOT_CSS: SystemStyleObject = {
  containerName: 'manager',
  containerType: 'inline-size',
  h: 'full',
  minH: '0',
  w: 'full',
};

/** Both panes stay mounted; the hidden one is visibility-hidden so scroll, virtual rows and form state survive. */
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

const ManagerBackContext = createContext<ManagerBack | null>(null);

const isShown = (element: Element): boolean => element.checkVisibility({ visibilityProperty: true });

/** Rows are found by the list's own state, never a remembered element: virtual rows are re-keyed and recycled. */
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

const findDetailEntry = (detail: HTMLElement): HTMLElement | null =>
  [...detail.querySelectorAll<HTMLElement>(BACK_SELECTOR)].find(isShown) ??
  [...detail.querySelectorAll<HTMLElement>('[role="tab"][aria-selected="true"]')].find(isShown) ??
  [...detail.querySelectorAll<HTMLElement>(TABBABLE_SELECTOR)].find(isShown) ??
  null;

export interface ManagerLayoutProps {
  library: ReactNode;
  detail: ReactNode;
  footer?: ReactNode;
  footerFills?: boolean;
  /** Which pane shows when only one fits; side by side it changes nothing. */
  isDetailOpen: boolean;
  backLabel: string;
  onBack: () => void;
}

/** Focus never stays on a hidden element: a resize or a popup refocusing a hidden trigger moves it to the pane. */
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

  /** A popup the switch came from may return focus to a hidden trigger, and the list may re-key the focused row. */
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

  // Before paint, so the ring never sits on a pane that just went away.
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

  // Every mounted manager listens; only the shown one has its search field in view. A modal suspends the key.
  useMountEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (gridRef.current && !isModalPresent(event)) {
        focusSearchOnSlash(event, gridRef.current, SEARCH_SELECTOR);
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => window.removeEventListener('keydown', handleKeyDown);
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

const SEARCH_ICON = <Icon as={SearchIcon} boxSize="3.5" color="fg.subtle" />;
const SEARCH_HINT = (
  <Kbd aria-hidden pointerEvents="none" variant="outline">
    /
  </Kbd>
);

export const ManagerSearchField = ({
  clearLabel,
  label,
  onChange,
  placeholder,
  size,
  value,
}: {
  clearLabel?: string;
  label: string;
  onChange: (value: string) => void;
  placeholder: string;
  size?: 'md' | 'lg';
  value: string;
}) => {
  const change = useCallback((event: ChangeEvent<HTMLInputElement>) => onChange(event.currentTarget.value), [onChange]);
  const clear = useCallback(() => onChange(''), [onChange]);
  const endElement = useMemo(() => {
    if (!value) {
      return SEARCH_HINT;
    }

    return clearLabel ? (
      <IconButton aria-label={clearLabel} me="-2" size="sm" tooltip={clearLabel} variant="ghost" onClick={clear}>
        <XIcon />
      </IconButton>
    ) : undefined;
  }, [clear, clearLabel, value]);

  return (
    <InputGroup endElement={endElement} startElement={SEARCH_ICON}>
      <Input
        aria-keyshortcuts="/"
        aria-label={label}
        data-manager-search=""
        placeholder={placeholder}
        size={size}
        value={value}
        onChange={change}
      />
    </InputGroup>
  );
};

/** Trimmed to cap height so flex centring centres the digits. */
const FILTER_COUNT_TEXT: SystemStyleObject = { textBox: 'trim-both cap alphabetic' };

/** The name ends with the count so the visible digits stay in it; sort order is not counted. */
export const ManagerFilterMenu = ({
  activeCount,
  children,
  label,
  ...rootProps
}: Omit<Menu.RootProps, 'children' | 'ids'> & {
  activeCount: number;
  children: ReactNode;
  label: string;
}) => {
  const { t } = useTranslation();
  const ids = useTooltipTriggerIds();
  const isFiltered = activeCount > 0;
  const name = isFiltered ? `${label}, ${t('common.activeFilterCount', { count: activeCount })}` : label;

  return (
    <Menu.Root ids={ids} {...rootProps}>
      <Tooltip content={name} ids={ids}>
        <Menu.Trigger asChild>
          <IconButton aria-label={name} color="fg.muted" flexShrink={0} variant="outline">
            <Icon as={SlidersHorizontalIcon} boxSize="4" />
            {isFiltered ? (
              <Box
                aria-hidden
                alignItems="center"
                bg="accent.solid"
                color="accent.contrast"
                display="flex"
                fontSize="xs"
                fontWeight="semibold"
                h="4"
                justifyContent="center"
                minW="4"
                outline="2px solid"
                outlineColor="bg"
                pointerEvents="none"
                position="absolute"
                px="1"
                right="-2.5"
                rounded="full"
                top="-2.5"
              >
                <Box as="span" css={FILTER_COUNT_TEXT}>
                  {activeCount}
                </Box>
              </Box>
            ) : null}
          </IconButton>
        </Menu.Trigger>
      </Tooltip>
      {children}
    </Menu.Root>
  );
};

/** Clears what `ManagerFilterMenu` counts; sort order and search stay. */
export const ManagerFilterResetItem = ({ activeCount, onReset }: { activeCount: number; onReset: () => void }) => {
  const { t } = useTranslation();

  return (
    <>
      <Menu.Separator />
      <MenuActionItem
        disabled={activeCount === 0}
        icon={RotateCcwIcon}
        label={t('common.resetFilters')}
        value="reset-filters"
        onSelect={onReset}
      />
    </>
  );
};

export interface ManagerAddAction {
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
  actions?: ReactNode;
  /** Shown only in single-pane mode, where the detail's tabs are out of reach. */
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

/** Mark the item tab `data-manager-item-tab` so it is the one that shrinks. */
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
