import type { ReactNode } from 'react';

import { Menu, Portal } from '@chakra-ui/react';
import { MenuContent } from '@workbench/components/ui';
import { useCallback, useMemo } from 'react';

import type { CanvasContextMenuTarget } from './canvasContextMenu';

export const CanvasGlobalContextMenu = ({
  children,
  onClose,
  target,
}: {
  children: ReactNode;
  onClose: () => void;
  target: CanvasContextMenuTarget;
}) => {
  const positioning = useMemo(
    () => ({
      getAnchorRect: () => ({ height: 1, width: 1, x: target.x, y: target.y }),
      placement: 'bottom-start' as const,
    }),
    [target.x, target.y]
  );
  const onOpenChange = useCallback(
    (details: { open: boolean }) => {
      if (!details.open) {
        onClose();
      }
    },
    [onClose]
  );

  return (
    <Menu.Root lazyMount open positioning={positioning} unmountOnExit onOpenChange={onOpenChange}>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="14rem" py="1">
            {children}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
