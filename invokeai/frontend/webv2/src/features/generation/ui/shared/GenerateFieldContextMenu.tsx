/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ReactNode } from 'react';

import { Box, Menu, Portal } from '@chakra-ui/react';
import { createExternalStore } from '@platform/state/externalStore';
import { MenuContent } from '@platform/ui/Menu';
import { useId } from 'react';
import { useTranslation } from 'react-i18next';

interface GenerateFieldContextMenuProps {
  children: ReactNode;
  /** The exact value put on the clipboard by Copy value. */
  copyValue: () => string;
  /** Disables the reset entry; the menu still opens for Copy value. */
  isAtDefault?: boolean;
  onReset?: () => void;
  /** Reset entry text; defaults to "Reset to model default". */
  resetLabel?: string;
}

/**
 * One field menu is open at a time. A right-click on a second field leaves the first menu open (right-button
 * presses don't dismiss it), and the two menus' dismissal then raced so the next right-click closed both; switching
 * owners closes the first menu instead.
 */
const openMenu = createExternalStore<{ owner: string | null; x: number; y: number }>({ owner: null, x: 0, y: 0 });

export const GenerateFieldContextMenu = ({
  children,
  copyValue,
  isAtDefault = false,
  onReset,
  resetLabel,
}: GenerateFieldContextMenuProps) => {
  const { t } = useTranslation();
  const owner = useId();
  const point = openMenu.useSelector((menu) => (menu.owner === owner ? { x: menu.x, y: menu.y } : null));

  return (
    <Box
      w="full"
      onContextMenu={(event) => {
        event.preventDefault();
        openMenu.setSnapshot({ owner, x: event.clientX, y: event.clientY });
      }}
    >
      {children}
      <Menu.Root
        open={point !== null}
        positioning={{
          getAnchorRect: () => (point ? { height: 1, width: 1, x: point.x, y: point.y } : null),
          placement: 'bottom-start',
        }}
        onOpenChange={(event) => {
          if (!event.open && openMenu.getSnapshot().owner === owner) {
            openMenu.patchSnapshot({ owner: null });
          }
        }}
      >
        <Portal>
          <Menu.Positioner>
            <MenuContent>
              {onReset ? (
                <Menu.Item disabled={isAtDefault} value="reset" onClick={onReset}>
                  {resetLabel ?? t('widgets.generate.resetToModelDefault')}
                </Menu.Item>
              ) : null}
              <Menu.Item value="copy" onClick={() => void navigator.clipboard.writeText(copyValue())}>
                {t('widgets.generate.copyValue')}
              </Menu.Item>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
    </Box>
  );
};
