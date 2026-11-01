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
}

const FIELD_CONTROL_SELECTOR =
  '[role="slider"]:not([aria-disabled="true"]), input:not([type="hidden"]):not(:disabled), button:not([data-modified-dot]):not(:disabled), [role="combobox"]';

/** The dot is drawn small; its button keeps a 16px target in the gutter left of the field. */
const MODIFIED_DOT_CSS = {
  '&::after': {
    bg: 'accent.solid',
    content: '""',
    h: '1.5',
    rounded: 'full',
    transitionDuration: 'var(--wb-motion-duration-fast)',
    transitionProperty: 'transform',
    w: '1.5',
  },
  '&:hover::after, &:focus-visible::after': { transform: 'scale(1.5)' },
  '&:focus-visible': { outline: '2px solid', outlineColor: 'accent.solid', outlineOffset: '-2px' },
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
            css={MODIFIED_DOT_CSS}
            cursor="pointer"
            display="flex"
            h="4"
            insetInlineStart="-3"
            justifyContent="center"
            position="absolute"
            rounded="full"
            top="1.5"
            type="button"
            w="4"
            onClick={resetFromDot}
          />
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
