import type { KeyboardEvent, MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent } from 'react';

import { Box, Icon, Menu, Portal, useMenu } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { ShortcutKeycaps } from '@workbench/hotkeys/keyGlyphs';
import { useCallback, useRef } from 'react';

const HOLD_OPEN_MS = 350;
const TOOLTIP_POSITIONING = { placement: 'right' } as const;
/** The gutter clears the strip's padding and border, so the menu opens beside the strip rather than over it. */
const MENU_POSITIONING = { gutter: 8, placement: 'right-start' } as const;
const MENU_ITEM_SELECTOR = '[role="menuitemradio"][data-value]';

export interface ToolFlyoutItem {
  id: string;
  icon: React.ElementType;
  label: string;
  /** The subtool's effective hotkey, already formatted for the platform. */
  shortcut?: readonly string[];
}

const isOverElement = (element: Element, event: ReactPointerEvent): boolean => {
  const rect = element.getBoundingClientRect();
  return (
    event.clientX >= rect.left &&
    event.clientX <= rect.right &&
    event.clientY >= rect.top &&
    event.clientY <= rect.bottom
  );
};

/** The menu entry under a pointer the slot has captured; capture retargets its events to the slot. */
const menuItemValueAt = (event: ReactPointerEvent): string | null =>
  document.elementFromPoint(event.clientX, event.clientY)?.closest<HTMLElement>(MENU_ITEM_SELECTOR)?.dataset.value ??
  null;

/**
 * A strip slot standing for a family of subtools. Click selects the current subtool; hold, right-click, ArrowRight
 * or the context-menu key open a menu of the family. Releasing a hold over an entry selects it; releasing elsewhere
 * leaves the menu open for a click.
 */
export const ToolFamilyButton = ({
  currentId,
  disabled = false,
  icon,
  isActive,
  items,
  label,
  onActivate,
  onSelectSubtool,
}: {
  /** The subtool the slot currently stands for (checked in the menu). */
  currentId: string;
  disabled?: boolean;
  icon: React.ElementType;
  isActive: boolean;
  items: readonly ToolFlyoutItem[];
  label: string;
  /** Plain click: select the current subtool. */
  onActivate: () => void;
  onSelectSubtool: (id: string) => void;
}) => {
  // The menu anchors to, and returns focus to, the element carrying the trigger id: the slot, through its tooltip.
  // The slot is not a `Menu.Trigger`, whose click would open the menu instead of selecting the current subtool.
  const ids = useTooltipTriggerIds();
  const menuMachine = useMenu({ ids, positioning: MENU_POSITIONING });
  const menu = menuMachine.api;
  const holdTimer = useRef<number | null>(null);
  const openedByHold = useRef(false);

  const clearHoldTimer = useCallback(() => {
    if (holdTimer.current !== null) {
      window.clearTimeout(holdTimer.current);
      holdTimer.current = null;
    }
  }, []);
  useMountEffect(() => clearHoldTimer);

  // Choosing an entry closes the menu itself; a locked slot's entries are disabled and never report a choice.
  const onValueChange = useCallback((details: { value: string }) => onSelectSubtool(details.value), [onSelectSubtool]);
  const openFromKeyboard = useCallback(() => {
    menu.setOpen(true);
    const first = items[0];
    if (first) {
      menu.setHighlightedValue(first.id);
    }
  }, [items, menu]);

  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLButtonElement>) => {
      if (disabled || event.button !== 0) {
        return;
      }
      openedByHold.current = false;
      clearHoldTimer();
      try {
        // Without capture the release over a menu entry targets the entry,
        // not this button, and the release-to-select path never runs.
        event.currentTarget.setPointerCapture(event.pointerId);
      } catch {
        // Synthetic pointers have no active id to capture.
      }
      holdTimer.current = window.setTimeout(() => {
        holdTimer.current = null;
        openedByHold.current = true;
        menu.setOpen(true);
      }, HOLD_OPEN_MS);
    },
    [clearHoldTimer, disabled, menu]
  );
  const onPointerUp = useCallback(
    (event: ReactPointerEvent<HTMLButtonElement>) => {
      if (disabled) {
        return;
      }
      if (holdTimer.current !== null) {
        clearHoldTimer();
        // Short press: an ordinary click on the slot — but only released over
        // it. Capture retargets the release here even after a drag-off, and a
        // drag-off release must cancel, like any button.
        if (!isOverElement(event.currentTarget, event)) {
          return;
        }
        if (menu.open) {
          menu.setOpen(false);
        } else {
          onActivate();
        }
        return;
      }
      if (menu.open && openedByHold.current) {
        openedByHold.current = false;
        // Held open: releasing over an entry selects it; elsewhere keeps the
        // menu open for a click.
        const value = menuItemValueAt(event);
        if (value !== null) {
          onSelectSubtool(value);
          menu.setOpen(false);
        }
      }
    },
    [clearHoldTimer, disabled, menu, onActivate, onSelectSubtool]
  );
  const onPointerMove = useCallback(
    (event: ReactPointerEvent<HTMLButtonElement>) => {
      if (holdTimer.current !== null) {
        // Leaving the slot cancels a pending hold; capture keeps delivering
        // moves here, so the bounds stand in for pointerleave.
        if (!isOverElement(event.currentTarget, event)) {
          clearHoldTimer();
        }
        return;
      }
      if (menu.open && openedByHold.current) {
        // Captured moves never reach the entries, so highlight the one the
        // drag is over the way hovering would.
        const value = menuItemValueAt(event);
        if (value !== null && value !== menu.highlightedValue) {
          menu.setHighlightedValue(value);
        }
      }
    },
    [clearHoldTimer, menu]
  );
  const onPointerCancel = useCallback(() => {
    // A canceled pointer (touch scroll takeover, OS gesture) must not open the
    // menu later; an already-open menu stays for a click, like quick release.
    clearHoldTimer();
    openedByHold.current = false;
  }, [clearHoldTimer]);
  const onClick = useCallback(
    (event: ReactMouseEvent<HTMLButtonElement>) => {
      // Pointer clicks are resolved on release above; a click with no pointer
      // detail is Enter or Space on the focused slot.
      if (event.detail === 0 && !disabled) {
        onActivate();
      }
    },
    [disabled, onActivate]
  );
  const onContextMenu = useCallback(
    (event: ReactMouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      if (!disabled) {
        menu.setOpen(true);
      }
    },
    [disabled, menu]
  );
  const onKeyDown = useCallback(
    (event: KeyboardEvent<HTMLButtonElement>) => {
      if (event.key === 'ArrowRight' || event.key === 'ContextMenu') {
        event.preventDefault();
        // Workbench hotkeys listen on window; the key that opens the menu must not also nudge the selected layer.
        event.stopPropagation();
        openFromKeyboard();
      }
    },
    [openFromKeyboard]
  );
  const onMenuKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      // The menu opens rightward from the strip, so ArrowLeft goes back to it.
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        menu.setOpen(false);
      }
    },
    [menu]
  );

  return (
    <Menu.RootProvider lazyMount unmountOnExit value={menuMachine}>
      {/* Forcing `open={false}` (not `disabled`) suppresses the tooltip while
          the menu is open: the disabled path unwraps the trigger and remounts
          the button, dropping pointer capture mid-hold. */}
      <Tooltip content={label} ids={ids} open={menu.open ? false : undefined} positioning={TOOLTIP_POSITIONING}>
        <IconButton
          aria-controls={menu.open ? menu.getContentProps().id : undefined}
          aria-expanded={menu.open}
          aria-haspopup="menu"
          aria-label={label}
          aria-pressed={isActive}
          disabled={disabled}
          variant={isActive ? 'solid' : 'ghost'}
          onClick={onClick}
          onContextMenu={onContextMenu}
          onKeyDown={onKeyDown}
          onPointerCancel={onPointerCancel}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
        >
          <Icon as={icon} boxSize="3.5" />
          {/* The corner tick: this slot holds more tools. */}
          <Box
            borderBottomColor="fg.subtle"
            borderBottomWidth="4px"
            borderLeftColor="transparent"
            borderLeftWidth="4px"
            bottom="0.5"
            h="0"
            pointerEvents="none"
            position="absolute"
            right="0.5"
            w="0"
          />
        </IconButton>
      </Tooltip>
      <Portal>
        <Menu.Positioner>
          <MenuContent aria-label={label} minW="10rem" onKeyDown={onMenuKeyDown}>
            <Menu.RadioItemGroup value={currentId} onValueChange={onValueChange}>
              {items.map((item) => (
                <Menu.RadioItem key={item.id} disabled={disabled} value={item.id}>
                  <Menu.ItemIndicator />
                  <Icon as={item.icon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
                  <Menu.ItemText fontSize="md">{item.label}</Menu.ItemText>
                  {item.shortcut && item.shortcut.length > 0 ? (
                    <Menu.ItemCommand>
                      <ShortcutKeycaps parts={item.shortcut} />
                    </Menu.ItemCommand>
                  ) : null}
                </Menu.RadioItem>
              ))}
            </Menu.RadioItemGroup>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.RootProvider>
  );
};
