/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { MouseEvent, ReactNode } from 'react';

import { Box, chakra, Menu, Portal } from '@chakra-ui/react';
import { canCopyText, copyText } from '@platform/browser/clipboard';
import { MenuContent, MenuItem, useContextMenu } from '@platform/ui/Menu';
import { Tooltip } from '@platform/ui/Tooltip';
import { useTranslation } from 'react-i18next';

interface GenerateFieldContextMenuProps {
  children: ReactNode;
  /** The exact value put on the clipboard by Copy value. */
  copyValue: () => string;
  /** The field's visible label, which names the modified dot ("Reset to model default: Steps"). */
  fieldLabel: string;
  /** Disables the reset entry and hides the modified dot; the menu still opens for Copy value. */
  isAtDefault?: boolean;
  onReset?: () => void;
  /** Reset entry text; defaults to "Reset to model default". */
  resetLabel?: string;
  /**
   * What the dot lines up with: a field's control (28px tall), or a switch row's first line (the switch's height).
   * Either way it stays in the gutter left of the row.
   */
  placement?: 'field' | 'switch';
}

/**
 * The dot's button, centred on the first line of what it marks: a control is 28px (`sizes.7`), and a switch row's first
 * line is the height of Generate's switch (16px, `sizes.4`), which its label's line height matches.
 *
 * Beside a field the button is a 16px target centred in the gutter; it may overlap the field's border. A switch row's
 * label is text from its first pixel and a click on it toggles, so there the button stays within the 8px gutter (8x16)
 * and sets the dot 1px further from the text: 4px at rest 3px clear of it, 6px on hover, and on focus the 4px dot with
 * its 1px ring, both ending at the gutter's outer edge.
 */
const DOT_BUTTON_PROPS = {
  field: {
    h: '4',
    insetInlineStart: '-3',
    justifyContent: 'center',
    top: 'calc(var(--chakra-sizes-7) / 2 - var(--chakra-sizes-2))',
    w: '4',
  },
  switch: {
    h: '4',
    insetInlineStart: '-2',
    justifyContent: 'flex-start',
    ps: '1px',
    top: 'calc(var(--chakra-sizes-4) / 2 - var(--chakra-sizes-2))',
    w: '2',
  },
} as const;

const FIELD_CONTROL_SELECTOR =
  '[role="slider"]:not([aria-disabled="true"]), input:not([type="hidden"]):not(:disabled), button:not([data-modified-dot]):not(:disabled), [role="combobox"]';

/**
 * Fields sit 8px in from the panel's edge, and the dot lives in that gutter: 4px at rest, 6px on hover or focus,
 * centred 4px left of the field so it keeps 1px clear of both the panel's edge and the field's border. Its button keeps
 * a 16px target, which may overlap both, so the button draws nothing itself: focus shows on the dot, as a 1px outline
 * that ends exactly at the gutter's edges (6px dot + 1px either side), where nothing clips it. An outline, not a
 * shadow, so it stays in forced colors.
 */
const MODIFIED_DOT_CSS = {
  '& > [data-part="dot"]': {
    bg: 'accent.solid',
    boxSize: '1',
    rounded: 'full',
    transitionDuration: 'var(--wb-motion-duration-fast)',
    transitionProperty: 'transform, width, height',
  },
  '&:hover:not(:focus-visible) > [data-part="dot"]': { transform: 'scale(1.5)' },
  '&:focus-visible': { outline: 'none' },
  '&:focus-visible > [data-part="dot"]': { boxSize: '1.5', outline: '1px solid', outlineColor: 'fg', outlineOffset: 0 },
  // On a switch row the focused dot keeps its 4px, so with its ring it fits the 7px between the text and the edge.
  '&[data-placement="switch"]:focus-visible > [data-part="dot"]': { boxSize: '1' },
} as const;

/**
 * A field's reset and copy menu. A changed field shows a dot in its gutter that resets it on click, so the reset is
 * visible without the menu; the menu opens on right-click or the keyboard menu key (Shift+F10), anchored to the
 * field when no pointer placed it.
 */
export const GenerateFieldContextMenu = ({
  children,
  copyValue,
  fieldLabel,
  isAtDefault = false,
  onReset,
  placement = 'field',
  resetLabel,
}: GenerateFieldContextMenuProps) => {
  const { t } = useTranslation();
  const { anchor: point, close, onRequestDismiss, open } = useContextMenu();
  const copyUnavailableReason = canCopyText() ? undefined : t('common.copyUnavailable');
  const resetText = resetLabel ?? t('widgets.generate.resetToModelDefault');
  const isModified = onReset !== undefined && !isAtDefault;
  const dotLabel = t('widgets.generate.resetFieldNamed', { field: fieldLabel, reset: resetText });

  // The dot unmounts once the field is back at its default; focus moves to the field rather than the page.
  const resetFromDot = (event: MouseEvent<HTMLButtonElement>) => {
    const wrapper = event.currentTarget.parentElement;

    onReset?.();
    requestAnimationFrame(() => wrapper?.querySelector<HTMLElement>(FIELD_CONTROL_SELECTOR)?.focus());
  };

  const handleContextMenu = (event: MouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    // A keyboard-opened menu has no pointer position of its own: Chromium reports button -1 at wherever the mouse last
    // was, other engines no pointer type or 0,0.
    const native = event.nativeEvent as Partial<PointerEvent>;
    const fromKeyboard =
      event.button === -1 || native.pointerType === '' || (event.clientX === 0 && event.clientY === 0);
    const focused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const rect = (
      focused && event.currentTarget.contains(focused) ? focused : event.currentTarget
    ).getBoundingClientRect();

    open({
      restoreFocus: focused ? () => focused.focus() : undefined,
      x: fromKeyboard ? rect.left : event.clientX,
      y: fromKeyboard ? rect.bottom : event.clientY,
    });
  };

  return (
    <Box position="relative" w="full" onContextMenu={handleContextMenu}>
      {/* Before the field in the DOM, as it sits left of it, so Tab reaches it first. */}
      {isModified ? (
        <Tooltip content={dotLabel}>
          <chakra.button
            alignItems="center"
            aria-label={dotLabel}
            data-modified-dot=""
            data-placement={placement}
            css={MODIFIED_DOT_CSS}
            cursor="pointer"
            display="flex"
            position="absolute"
            rounded="full"
            type="button"
            {...DOT_BUTTON_PROPS[placement]}
            onClick={resetFromDot}
          >
            <chakra.span aria-hidden data-part="dot" />
          </chakra.button>
        </Tooltip>
      ) : null}
      {children}
      <Menu.Root
        open={point !== null}
        positioning={{
          getAnchorRect: () => (point ? { height: 1, width: 1, x: point.x, y: point.y } : null),
          placement: 'bottom-start',
        }}
        onOpenChange={(event) => {
          if (!event.open) {
            close();
          }
        }}
        onRequestDismiss={onRequestDismiss}
      >
        <Portal>
          <Menu.Positioner>
            <MenuContent>
              {onReset ? (
                <Menu.Item disabled={isAtDefault} value="reset" onClick={onReset}>
                  {resetText}
                </Menu.Item>
              ) : null}
              <MenuItem
                unavailableReason={copyUnavailableReason}
                value="copy"
                onClick={() => void copyText(copyValue()).catch(() => undefined)}
              >
                {t('widgets.generate.copyValue')}
              </MenuItem>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
    </Box>
  );
};
