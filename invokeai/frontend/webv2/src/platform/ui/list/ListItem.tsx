import type { MouseEvent, PointerEvent, ReactNode } from 'react';

import { Checkbox, chakra, useSlotRecipe } from '@chakra-ui/react';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { listItemSlotRecipe } from '@theme/recipes';
import { useCallback, useId, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

export type ListDensity = 'compact' | 'regular' | 'comfortable' | 'snug';

export interface ListContextMenuAnchor {
  x: number;
  y: number;
  /** Resolved late because virtualization or the action can unmount the row; suits a Dialog's `finalFocusEl`. */
  focusTarget: () => HTMLElement | null;
  restoreFocus: () => void;
}

export interface ListItemProps {
  title: string;
  /** While set, the primary area is no button and the row opens no context menu; the caller ends the edit. */
  titleEditor?: ReactNode;
  /** Identifiers keep their tail visible; prose keeps its start. */
  titleTruncate?: 'middle' | 'end';
  /** Media or icon before the text; size it for the density (36px at comfortable and snug). */
  leading?: ReactNode;
  badges?: ReactNode;
  description?: ReactNode;
  trailing?: ReactNode;
  actions?: ReactNode;
  /** Sits outside the primary button, so it may hold controls; its own context menu takes precedence. */
  detail?: ReactNode;
  density?: ListDensity;
  isActive?: boolean;
  isChecked?: boolean;
  /** Shown but inert while its page is being replaced; focus is kept so paging does not lose it. */
  isBusy?: boolean;
  /** Only the primary button stays focusable, so arrows reach it and the reason is read; all else is off. */
  isDisabled?: boolean;
  isExpanded?: boolean;
  isMenuOpen?: boolean;
  checkLabel?: string;
  itemKey?: string;
  id?: string;
  /** Named only while expanded: a modal's focus trap admits portalled content only an expanded control names. */
  controls?: string;
  /** `presentation` when a caller's own list item wraps this row together with content that belongs to it. */
  role?: 'listitem' | 'presentation';
  tabIndex?: number;
  positionInSet?: number;
  setSize?: number;
  onPress?: (event: MouseEvent<HTMLButtonElement>) => void;
  pressLabel?: string;
  onIntent?: () => void;
  onCheckedChange?: (checked: boolean) => void;
  /** Without onPress, activation also opens it, except from a mouse click (right-click is that path). */
  onContextMenu?: (anchor: ListContextMenuAnchor) => void;
}

const PRIMARY_SELECTOR = '[data-list-primary]';
const TRUNCATE_CSS = { display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } as const;

/** Checkbox and actions are siblings of the primary button, so nothing nests an interactive control. */
export const ListItem = ({
  actions,
  badges,
  checkLabel,
  controls,
  density = 'regular',
  description,
  detail,
  id,
  isActive = false,
  isBusy = false,
  isChecked = false,
  isDisabled = false,
  isExpanded,
  isMenuOpen = false,
  itemKey,
  leading,
  positionInSet,
  role = 'listitem',
  setSize,
  tabIndex = 0,
  title,
  titleEditor,
  titleTruncate = 'middle',
  trailing,
  onCheckedChange,
  onContextMenu,
  onIntent,
  onPress,
  pressLabel,
}: ListItemProps) => {
  // Own ids, so a row inside a form Field never points the checkbox's label at the Field's input.
  const checkInputId = useId();
  const checkIds = useMemo(() => ({ hiddenInput: checkInputId }), [checkInputId]);
  const { t } = useTranslation();
  const recipe = useSlotRecipe({ recipe: listItemSlotRecipe });
  const tone = isActive ? 'accent' : isChecked ? 'selected' : 'none';
  const styles = useMemo(() => recipe({ active: tone, density }), [density, recipe, tone]);
  const isInteractive = onPress !== undefined || onContextMenu !== undefined || itemKey !== undefined;
  const isStatic = !isInteractive && (actions === undefined || actions === null);
  const isEditingTitle = titleEditor !== undefined && titleEditor !== null;
  const menu = isDisabled || isEditingTitle ? undefined : onContextMenu;
  const intent = isDisabled ? undefined : onIntent;
  const isPrimaryExpanded = menu && !onPress ? isMenuOpen : isExpanded;

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
  // Detect a mouse press positively: screen readers synthesize pointer-like clicks, and touch has no right-click.
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
      if (isBusy || isDisabled) {
        return;
      }
      if (onPress) {
        onPress(event);
      } else if (!isMouseClick && event.currentTarget.parentElement) {
        openContextMenu(event.currentTarget.parentElement);
      }
    },
    [isBusy, isDisabled, onPress, openContextMenu]
  );
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
          {isEditingTitle ? (
            titleEditor
          ) : (
            <MiddleTruncate
              as="span"
              css={styles.title}
              tailGraphemes={titleTruncate === 'end' ? 0 : undefined}
              text={title}
            />
          )}
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
      data-disabled={isDisabled || undefined}
      data-list-row={itemKey}
      data-list-surface=""
      data-menu-open={isMenuOpen || undefined}
      data-static={isStatic || undefined}
      role={role}
      onContextMenu={menu ? handleContextMenu : undefined}
    >
      {onCheckedChange ? (
        <chakra.div css={styles.check}>
          <Checkbox.Root
            aria-label={checkLabel ?? t('common.list.selectItem', { name: title })}
            checked={isChecked}
            ids={checkIds}
            colorPalette="accent"
            disabled={isBusy || isDisabled}
            size="sm"
            onCheckedChange={handleCheckedChange}
          >
            <Checkbox.HiddenInput />
            <Checkbox.Control />
          </Checkbox.Root>
        </chakra.div>
      ) : null}
      {isInteractive && !isEditingTitle ? (
        <chakra.button
          aria-controls={isPrimaryExpanded ? controls : undefined}
          aria-current={isActive || undefined}
          aria-disabled={isBusy || isDisabled || undefined}
          aria-expanded={isPrimaryExpanded}
          aria-haspopup={menu && !onPress ? 'menu' : undefined}
          aria-label={pressLabel}
          css={styles.primary}
          data-list-primary=""
          id={id}
          tabIndex={tabIndex}
          type="button"
          onClick={handlePress}
          onFocus={intent}
          onPointerDown={onPress ? undefined : handlePointerDown}
          onPointerEnter={intent}
        >
          {content}
        </chakra.button>
      ) : (
        // Static rows (nothing to press, no menu) and a row editing its title must not announce themselves as buttons.
        <chakra.div aria-current={isActive || undefined} css={styles.primary} data-list-primary="">
          {content}
        </chakra.div>
      )}
      {actions !== undefined && actions !== null ? (
        <chakra.div css={styles.actions} inert={isDisabled || undefined}>
          {actions}
        </chakra.div>
      ) : null}
      {detail !== undefined && detail !== null ? (
        <chakra.div css={styles.detail} inert={isDisabled || undefined}>
          {detail}
        </chakra.div>
      ) : null}
    </chakra.div>
  );
};
