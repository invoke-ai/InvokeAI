import type { CanvasDocumentContractV3, LayerStackKind } from '@workbench/canvas-engine/api';
import type { ComponentProps } from 'react';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { MenuContent } from '@platform/ui';
import { useCallback, useMemo } from 'react';

import type { LayerSurfaceAnchor } from './layerRowCommands';

import { useLayerStackActions, type LayerStackActionsEngine } from './useLayerStackActions';

type MenuPositioning = ComponentProps<typeof Menu.Root>['positioning'];

interface LayerStackMenuProps {
  anchor: LayerSurfaceAnchor;
  document: CanvasDocumentContractV3;
  editingLocked: boolean;
  engine: LayerStackActionsEngine | null;
  stack: LayerStackKind;
  onClose: () => void;
}

/** The stack header's actions as a menu, so the keyboard reaches them from the header tree item. */
export const LayerStackMenu = ({ anchor, document, editingLocked, engine, stack, onClose }: LayerStackMenuProps) => {
  const actions = useLayerStackActions(stack, document, engine, editingLocked);
  const positioning = useMemo<MenuPositioning>(
    () => ({ getAnchorRect: () => anchor, placement: 'bottom-start' }),
    [anchor]
  );
  const handleOpenChange = useCallback(
    (details: { open: boolean }) => {
      if (!details.open) {
        onClose();
      }
    },
    [onClose]
  );
  return (
    <Menu.Root lazyMount open positioning={positioning} unmountOnExit onOpenChange={handleOpenChange}>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="13rem" py="1">
            {actions.map((action) => (
              <Menu.Item key={action.id} disabled={action.disabled} value={action.id} onSelect={action.run}>
                <HStack gap="2" minW="0" w="full">
                  <Icon as={action.icon} boxSize="3.5" color="fg.muted" flexShrink={0} />
                  <Text flex="1" fontSize="xs">
                    {action.label}
                  </Text>
                </HStack>
              </Menu.Item>
            ))}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
