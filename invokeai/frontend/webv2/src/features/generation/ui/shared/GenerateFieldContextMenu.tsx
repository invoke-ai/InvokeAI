/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ReactNode } from 'react';

import { Box, Menu, Portal } from '@chakra-ui/react';
import { MenuContent } from '@platform/ui/Menu';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

interface GenerateFieldContextMenuProps {
  children: ReactNode;
  /** The exact value put on the clipboard by Copy value. */
  copyValue: () => string;
  /** Disables the reset entry; the menu still opens for Copy value. */
  isAtDefault?: boolean;
  onReset?: () => void;
}

/**
 * Desktop-grade right-click on a model-governed field: Reset to model default
 * and Copy value, anchored at the pointer. Wraps the field without affecting
 * its layout; left-click behavior is untouched.
 */
export const GenerateFieldContextMenu = ({
  children,
  copyValue,
  isAtDefault = false,
  onReset,
}: GenerateFieldContextMenuProps) => {
  const { t } = useTranslation();
  const [point, setPoint] = useState<{ x: number; y: number } | null>(null);

  return (
    <Box
      w="full"
      onContextMenu={(event) => {
        event.preventDefault();
        setPoint({ x: event.clientX, y: event.clientY });
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
          if (!event.open) {
            setPoint(null);
          }
        }}
      >
        <Portal>
          <Menu.Positioner>
            <MenuContent>
              {onReset ? (
                <Menu.Item disabled={isAtDefault} value="reset" onClick={onReset}>
                  {t('widgets.generate.resetToModelDefault')}
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
