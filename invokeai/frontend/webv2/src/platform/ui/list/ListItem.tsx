import type { MouseEvent, PointerEvent, ReactNode } from 'react';

import { Checkbox, chakra, useSlotRecipe } from '@chakra-ui/react';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { listItemSlotRecipe } from '@theme/recipes';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

export type ListDensity = 'compact' | 'regular' | 'comfortable' | 'snug';

export interface ListContextMenuAnchor {
  x: number;
  y: number;
  /**
   * Where keyboard focus belongs once the menu, or a dialog it opened, closes: this row while it exists, otherwise
   * the list's current tab stop. Resolved late because virtualization or the action itself can unmount the row;
   * suits a Dialog's `finalFocusEl`.
   */
  focusTarget: () => HTMLElement | null;
  restoreFocus: () => void;
}

export interface ListItemProps {
  title: string;
  /** Identifiers keep their tail visible; prose keeps its start. */
  titleTruncate?: 'middle' | 'end';
  /** Media or icon before the text; size it for the density (36px at comfortable and snug). */
  leading?: ReactNode;
  /** Inline after the title, e.g. a status badge; keep it short so the title keeps its room. */
  badges?: ReactNode;
  /** Second line: muted text, or a node such as a badge row. */
  description?: ReactNode;
  /** Trailing metadata or badges; string content renders muted. */
  trailing?: ReactNode;
  /** Controls that act on the row, rendered beside the primary button so they never nest inside it. */
  actions?: ReactNode;
  /**
   * Content under the row that belongs to it, such as a control tuning the item; it shares the row's surface but sits
   * outside the primary button, so it may hold controls. A context menu it handles itself takes precedence.
   */
  detail?: ReactNode;
  density?: ListDensity;
  /** The one row whose detail is open; announced with aria-current and filled with the accent tone. */
  isActive?: boolean;
  /** Part of a multi-selection; only meaningful with onCheckedChange. */
  isChecked?: boolean;
  /** Shown but inert while its page is being replaced; focus is kept so paging does not lose it. */
  isBusy?: boolean;
  /** The row discloses content below itself; announced with aria-expanded. */
  isExpanded?: boolean;
  /** This row's context menu is open; the row keeps its hover surface so it is clear what the menu acts on. */
  isMenuOpen?: boolean;
  /** Accessible name of the checkbox when the title alone is ambiguous; defaults to "Select {title}". */
  checkLabel?: string;
  /** Identifies the row to its owning List for focus management. */
  itemKey?: string;
  /** `presentation` when a caller's own list item wraps this row together with content that belongs to it. */
  role?: 'listitem' | 'presentation';
  /** Roving tabindex from the owning List; standalone rows are tab stops. */
  tabIndex?: number;
  positionInSet?: number;
  setSize?: number;
  /** Makes a standalone row a button; without it, and without a context menu, the row is static content. */
  onPress?: (event: MouseEvent<HTMLButtonElement>) => void;
  /** Fires on hover and focus, ahead of a press, to warm what the press will need. */
  onIntent?: () => void;
  /** Renders the leading checkbox. */
  onCheckedChange?: (checked: boolean) => void;
  /**
   * Without onPress, activating the primary button also opens this menu, except from a mouse click, where
   * right-click is the path; keyboard, assistive-technology, touch, and pen activation all open it.
   */
  onContextMenu?: (anchor: ListContextMenuAnchor) => void;
}

const PRIMARY_SELECTOR = '[data-list-primary]';
const TRUNCATE_CSS = { display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } as const;

/**
 * The checkbox and the actions are siblings of the primary button so nothing nests an interactive control. The
 * whole surface hovers as one; the primary button carries the accessible name and the focus ring wraps the row.
 */
export const ListItem = ({
  actions,
  badges,
  checkLabel,
  density = 'regular',
  description,
  detail,
  isActive = false,
  isBusy = false,
  isChecked = false,
  isExpanded,
  isMenuOpen = false,
  itemKey,
  leading,
  positionInSet,
  role = 'listitem',
  setSize,
  tabIndex = 0,
  title,
  titleTruncate = 'middle',
  trailing,
  onCheckedChange,
  onContextMenu,
  onIntent,
  onPress,
}: ListItemProps) => {
  const { t } = useTranslation();
  const recipe = useSlotRecipe({ recipe: listItemSlotRecipe });
  const tone = isActive ? 'accent' : isChecked ? 'selected' : 'none';
  const styles = useMemo(() => recipe({ active: tone, density }), [density, recipe, tone]);
  // Inside a List every row is a keyboard stop; a standalone row is a button only when pressing it does something.
  const isInteractive = onPress !== undefined || onContextMenu !== undefined || itemKey !== undefined;
  // A row with actions is pointed like an interactive one, so its buttons read as belonging to it; only a row with
  // nothing to act on stays static.
  const isStatic = !isInteractive && (actions === undefined || actions === null);

  const openContextMenu = useCallback(
    (row: HTMLElement, point?: { x: number; y: number }) => {
      if (!onContextMenu) {
        return;
      }

      const viewport = row.closest<HTMLElement>('[data-list-viewport]');
      const key = row.getAttribute('data-list-row');
      const rect = row.getBoundingClientRect();
      const focusTarget = (): HTMLElement | null => {
        if (row.isConnected) {
          return row.querySelector<HTMLElement>(PRIMARY_SELECTOR);
        }

        if (!viewport) {
          return null;
        }

        const byKey =
          key === null
            ? null
            : viewport.querySelector<HTMLElement>(`[data-list-row="${CSS.escape(key)}"] ${PRIMARY_SELECTOR}`);

        return byKey ?? viewport.querySelector<HTMLElement>(`${PRIMARY_SELECTOR}[tabindex="0"]`);
      };

      onContextMenu({
        focusTarget,
        restoreFocus: () => focusTarget()?.focus(),
        x: point?.x ?? rect.left + 8,
        y: point?.y ?? rect.bottom,
      });
    },
    [onContextMenu]
  );
  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      // A nested field's menu takes precedence. Keyboard-invoked menus can report (0,0).
      if (event.defaultPrevented) {
        return;
      }
      event.preventDefault();
      openContextMenu(
        event.currentTarget,
        event.clientX === 0 && event.clientY === 0 ? undefined : { x: event.clientX, y: event.clientY }
      );
    },
    [openContextMenu]
  );
  const handleCheckedChange = useCallback(
    (details: { checked: boolean | 'indeterminate' }) => onCheckedChange?.(details.checked === true),
    [onCheckedChange]
  );
  // Detect a mouse press positively: screen readers synthesize clicks that look like pointer clicks (detail 1), and
  // touch has no right-click, so only a click that a mouse pressed skips the menu.
  const pressPointerType = useRef<string | null>(null);
  const handlePointerDown = useCallback((event: PointerEvent<HTMLButtonElement>) => {
    pressPointerType.current = event.pointerType;
  }, []);
  const handlePress = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      // Firefox marks screen-reader activation as a virtual input source even when it fires pointer events for it.
      const isVirtualClick = (event.nativeEvent as { mozInputSource?: number }).mozInputSource === 0;
      const isMouseClick = pressPointerType.current === 'mouse' && event.detail > 0 && !isVirtualClick;
      pressPointerType.current = null;
      if (isBusy) {
        return;
      }
      if (onPress) {
        onPress(event);
      } else if (!isMouseClick && event.currentTarget.parentElement) {
        openContextMenu(event.currentTarget.parentElement);
      }
    },
    [isBusy, onPress, openContextMenu]
  );
  // Plain text keeps to its line; a node (badge row, figures) lays itself out.
  const descriptionCss = useMemo(
    () => (typeof description === 'string' ? [styles.description, TRUNCATE_CSS] : styles.description),
    [description, styles.description]
  );
  const trailingCss = useMemo(
    () => (typeof trailing === 'string' ? [styles.trailing, TRUNCATE_CSS] : styles.trailing),
    [styles.trailing, trailing]
  );

  const content = (
    <>
      {leading}
      <chakra.span css={styles.body}>
        <chakra.span css={styles.titleLine}>
          <MiddleTruncate
            as="span"
            css={styles.title}
            tailGraphemes={titleTruncate === 'end' ? 0 : undefined}
            text={title}
          />
          {badges !== undefined && badges !== null ? <chakra.span css={styles.badges}>{badges}</chakra.span> : null}
        </chakra.span>
        {description !== undefined && description !== null ? (
          <chakra.span css={descriptionCss}>{description}</chakra.span>
        ) : null}
      </chakra.span>
      {trailing !== undefined && trailing !== null ? <chakra.span css={trailingCss}>{trailing}</chakra.span> : null}
    </>
  );

  return (
    <chakra.div
      aria-posinset={positionInSet}
      aria-setsize={setSize}
      css={styles.root}
      data-busy={isBusy || undefined}
      data-list-row={itemKey}
      data-list-surface=""
      data-menu-open={isMenuOpen || undefined}
      data-static={isStatic || undefined}
      role={role}
      onContextMenu={onContextMenu ? handleContextMenu : undefined}
    >
      {onCheckedChange ? (
        <chakra.div css={styles.check}>
          <Checkbox.Root
            aria-label={checkLabel ?? t('common.list.selectItem', { name: title })}
            checked={isChecked}
            colorPalette="accent"
            disabled={isBusy}
            size="sm"
            onCheckedChange={handleCheckedChange}
          >
            <Checkbox.HiddenInput />
            <Checkbox.Control />
          </Checkbox.Root>
        </chakra.div>
      ) : null}
      {isInteractive ? (
        <chakra.button
          aria-current={isActive || undefined}
          aria-disabled={isBusy || undefined}
          aria-expanded={onContextMenu && !onPress ? isMenuOpen : isExpanded}
          aria-haspopup={onContextMenu && !onPress ? 'menu' : undefined}
          css={styles.primary}
          data-list-primary=""
          tabIndex={tabIndex}
          type="button"
          onClick={handlePress}
          onFocus={onIntent}
          onPointerDown={onPress ? undefined : handlePointerDown}
          onPointerEnter={onIntent}
        >
          {content}
        </chakra.button>
      ) : (
        // Static rows (nothing to press, no menu) must not announce themselves as buttons.
        <chakra.div aria-current={isActive || undefined} css={styles.primary} data-list-primary="">
          {content}
        </chakra.div>
      )}
      {actions !== undefined && actions !== null ? <chakra.div css={styles.actions}>{actions}</chakra.div> : null}
      {detail !== undefined && detail !== null ? <chakra.div css={styles.detail}>{detail}</chakra.div> : null}
    </chakra.div>
  );
};
