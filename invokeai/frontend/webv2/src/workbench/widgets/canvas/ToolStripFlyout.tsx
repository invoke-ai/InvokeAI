import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';

import { Box, Icon } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { useCallback, useEffect, useRef, useState } from 'react';

const HOLD_OPEN_MS = 350;
const TOOLTIP_POSITIONING = { placement: 'right' } as const;
/** The flyout opens sideways, so a subtool's tooltip goes below it rather than over the next subtool. */
const SUBTOOL_TOOLTIP_POSITIONING = { placement: 'bottom' } as const;

export interface ToolFlyoutItem {
  id: string;
  icon: React.ElementType;
  label: string;
}

/**
 * Click selects the current subtool; hold, right-click, or keyboard opens the flyout. Release over an entry
 * selects; quick release leaves it open.
 */
export const ToolFamilyButton = ({
  currentId,
  disabled,
  icon,
  isActive,
  items,
  label,
  onActivate,
  onSelectSubtool,
}: {
  /** The subtool the slot currently stands for (checked in the flyout). */
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
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const holdTimer = useRef<number | null>(null);
  const openedByHold = useRef(false);

  const clearHoldTimer = useCallback(() => {
    if (holdTimer.current !== null) {
      window.clearTimeout(holdTimer.current);
      holdTimer.current = null;
    }
  }, []);
  const close = useCallback(() => {
    setOpen(false);
    openedByHold.current = false;
  }, []);

  // Outside pointers and Escape close the flyout; the listener lives only while open.
  useEffect(() => {
    if (!open) {
      return;
    }
    const onOutside = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        close();
      }
    };
    const onEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') {
        close();
        buttonRef.current?.focus();
      }
    };
    window.addEventListener('pointerdown', onOutside, true);
    window.addEventListener('keydown', onEscape, true);
    return () => {
      window.removeEventListener('pointerdown', onOutside, true);
      window.removeEventListener('keydown', onEscape, true);
    };
  }, [close, open]);
  useEffect(() => clearHoldTimer, [clearHoldTimer]);

  const select = useCallback(
    (id: string) => {
      // The focused entry unmounts with the flyout; keep focus in the strip.
      const restoreFocus = rootRef.current?.contains(document.activeElement) ?? false;
      onSelectSubtool(id);
      close();
      if (restoreFocus) {
        buttonRef.current?.focus();
      }
    },
    [close, onSelectSubtool]
  );
  const onMenuKeyDown = useCallback((event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft' && event.key !== 'Home' && event.key !== 'End') {
      return;
    }
    const entries = Array.from(rootRef.current?.querySelectorAll<HTMLElement>('[data-subtool-id]') ?? []);
    if (entries.length === 0) {
      return;
    }
    event.preventDefault();
    const index = entries.indexOf(document.activeElement as HTMLElement);
    const next =
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? entries.length - 1
          : (index + (event.key === 'ArrowRight' ? 1 : -1) + entries.length) % entries.length;
    entries[next]?.focus();
  }, []);

  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLButtonElement>) => {
      if (disabled || event.button !== 0) {
        return;
      }
      openedByHold.current = false;
      clearHoldTimer();
      try {
        // Without capture the release over a flyout entry targets the entry,
        // not this button, and the release-to-select path never runs.
        event.currentTarget.setPointerCapture(event.pointerId);
      } catch {
        // Synthetic pointers have no active id to capture.
      }
      holdTimer.current = window.setTimeout(() => {
        holdTimer.current = null;
        openedByHold.current = true;
        setOpen(true);
      }, HOLD_OPEN_MS);
    },
    [clearHoldTimer, disabled]
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
        const rect = event.currentTarget.getBoundingClientRect();
        const overButton =
          event.clientX >= rect.left &&
          event.clientX <= rect.right &&
          event.clientY >= rect.top &&
          event.clientY <= rect.bottom;
        if (!overButton) {
          return;
        }
        if (!open) {
          onActivate();
        } else {
          close();
        }
        return;
      }
      if (open && openedByHold.current) {
        // Held open: releasing over an entry selects it; elsewhere keeps the
        // flyout open for a click.
        const under = document.elementFromPoint(event.clientX, event.clientY);
        const item = under?.closest<HTMLElement>('[data-subtool-id]');
        if (item?.dataset.subtoolId) {
          select(item.dataset.subtoolId);
        }
        openedByHold.current = false;
      }
    },
    [clearHoldTimer, close, disabled, onActivate, open, select]
  );
  const onPointerMove = useCallback(
    (event: ReactPointerEvent<HTMLButtonElement>) => {
      // Cancel pending hold when the pointer leaves, but preserve open flyouts; capture requires bounds checks
      // instead of pointerleave.
      if (holdTimer.current === null) {
        return;
      }
      const rect = event.currentTarget.getBoundingClientRect();
      if (
        event.clientX < rect.left ||
        event.clientX > rect.right ||
        event.clientY < rect.top ||
        event.clientY > rect.bottom
      ) {
        clearHoldTimer();
      }
    },
    [clearHoldTimer]
  );
  const onPointerCancel = useCallback(() => {
    // A canceled pointer (touch scroll takeover, OS gesture) must not open the
    // flyout later; an already-open flyout stays for a click, like quick release.
    clearHoldTimer();
    openedByHold.current = false;
  }, [clearHoldTimer]);
  const onContextMenu = useCallback(
    (event: ReactPointerEvent<HTMLButtonElement> | React.MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      if (!disabled) {
        setOpen(true);
      }
    },
    [disabled]
  );
  const onKeyDown = useCallback((event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === 'ArrowRight' || event.key === 'ContextMenu') {
      event.preventDefault();
      setOpen(true);
      requestAnimationFrame(() => {
        rootRef.current?.querySelector<HTMLElement>('[data-subtool-id]')?.focus();
      });
    }
  }, []);

  return (
    <Box ref={rootRef} position="relative">
      {/* Forcing `open={false}` (not `disabled`) suppresses the tooltip while
          the flyout sits in its spot: the disabled path unwraps the trigger and
          remounts the button, dropping pointer capture mid-hold. */}
      <Tooltip content={label} open={open ? false : undefined} positioning={TOOLTIP_POSITIONING}>
        <IconButton
          ref={buttonRef}
          aria-expanded={open}
          aria-haspopup="menu"
          aria-label={label}
          aria-pressed={isActive}
          disabled={disabled}
          size="xs"
          variant={isActive ? 'solid' : 'ghost'}
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
      {open ? (
        <Box
          aria-label={label}
          bg="bg.panel"
          borderColor="border.subtle"
          borderWidth="1px"
          display="flex"
          gap="0.5"
          left="calc(100% + 4px)"
          p="0.5"
          position="absolute"
          role="menu"
          rounded="md"
          shadow="md"
          top="0"
          zIndex="3"
          onKeyDown={onMenuKeyDown}
        >
          {items.map((item) => (
            <FlyoutItem key={item.id} checked={item.id === currentId} item={item} onSelect={select} />
          ))}
        </Box>
      ) : null}
    </Box>
  );
};

const FlyoutItem = ({
  checked,
  item,
  onSelect,
}: {
  checked: boolean;
  item: ToolFlyoutItem;
  onSelect: (id: string) => void;
}) => {
  const onClick = useCallback(() => onSelect(item.id), [item.id, onSelect]);
  return (
    <Tooltip content={item.label} positioning={SUBTOOL_TOOLTIP_POSITIONING}>
      <IconButton
        aria-checked={checked}
        aria-label={item.label}
        data-subtool-id={item.id}
        role="menuitemradio"
        size="xs"
        variant={checked ? 'solid' : 'ghost'}
        onClick={onClick}
      >
        <Icon as={item.icon} boxSize="3.5" />
      </IconButton>
    </Tooltip>
  );
};
