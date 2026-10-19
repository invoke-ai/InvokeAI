import type { ComponentProps } from 'react';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { MenuActionItem, MenuContent } from '@platform/ui';
import {
  ArrowDownIcon,
  ArrowRightIcon,
  ArrowUpIcon,
  ChevronRightIcon,
  CircleIcon,
  CircleOffIcon,
  CopyIcon,
  PencilIcon,
  XIcon,
} from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { LayerRowCommands, LayerSurfaceAnchor } from './layerRowCommands';

import { isRenameableChildKind } from './LayerChildRow';
import {
  isOrderedChildKind,
  layerChildRemoveLabelKey,
  layerChildRenameLabelKey,
  type ProjectedChildRow,
} from './layerChildRows';

const ChildMenuItem = MenuActionItem;

type MenuPositioning = ComponentProps<typeof Menu.Root>['positioning'];

const SUBMENU_POSITIONING = { placement: 'right-start' } as const;
const noop = (): void => undefined;

/** The context menu of a projected child row: toggle, rename, reorder, move to another layer, or remove it. */
export const LayerChildMenu = ({
  anchor,
  child,
  commands,
  editingLocked,
  moveTargets,
  onClose,
}: {
  anchor: LayerSurfaceAnchor;
  child: ProjectedChildRow;
  commands: LayerRowCommands;
  editingLocked: boolean;
  /** Other layers the item can move to, keyboard parity for the cross-layer drag. */
  moveTargets: readonly { id: string; name: string }[];
  onClose: () => void;
}) => {
  const { t } = useTranslation();
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
  const handleToggle = useCallback(() => commands.setChildEnabled(child, !child.isEnabled), [child, commands]);
  const handleRemove = useCallback(() => commands.removeChild(child), [child, commands]);
  const handleDuplicate = useCallback(() => commands.duplicateChild(child), [child, commands]);
  const handleRename = useCallback(() => commands.startRename(child.key), [child.key, commands]);
  const handleMoveUp = useCallback(() => commands.moveChild(child, -1), [child, commands]);
  const handleMoveDown = useCallback(() => commands.moveChild(child, 1), [child, commands]);
  const ordered = isOrderedChildKind(child.kind);

  return (
    <Menu.Root lazyMount open positioning={positioning} unmountOnExit onOpenChange={handleOpenChange}>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="10rem">
            <ChildMenuItem
              disabled={editingLocked}
              icon={child.isEnabled ? CircleOffIcon : CircleIcon}
              label={t(child.isEnabled ? 'widgets.layers.modifiers.disable' : 'widgets.layers.modifiers.enable')}
              value="toggle"
              onSelect={handleToggle}
            />
            {isRenameableChildKind(child.kind) ? (
              <ChildMenuItem
                disabled={editingLocked}
                icon={PencilIcon}
                label={t(layerChildRenameLabelKey(child.kind))}
                value="rename"
                onSelect={handleRename}
              />
            ) : null}
            {ordered ? (
              <>
                <ChildMenuItem
                  disabled={editingLocked}
                  icon={CopyIcon}
                  label={t('widgets.layers.modifiers.duplicateAdjustment')}
                  value="duplicate"
                  onSelect={handleDuplicate}
                />
                <ChildMenuItem
                  disabled={editingLocked || (child.orderedPosInSet ?? 1) <= 1}
                  icon={ArrowUpIcon}
                  label={t('widgets.layers.modifiers.moveUp')}
                  value="move-up"
                  onSelect={handleMoveUp}
                />
                <ChildMenuItem
                  disabled={editingLocked || (child.orderedPosInSet ?? 0) >= (child.orderedSetSize ?? 0)}
                  icon={ArrowDownIcon}
                  label={t('widgets.layers.modifiers.moveDown')}
                  value="move-down"
                  onSelect={handleMoveDown}
                />
              </>
            ) : null}
            {moveTargets.length === 0 ? null : editingLocked ? (
              <ChildMenuItem
                disabled
                icon={ArrowRightIcon}
                label={t('widgets.layers.modifiers.moveToLayer')}
                value="move-to"
                onSelect={noop}
              />
            ) : (
              <Menu.Root positioning={SUBMENU_POSITIONING}>
                <Menu.TriggerItem aria-label={t('widgets.layers.modifiers.moveToLayer')}>
                  <HStack gap="2" minW="0" w="full">
                    <Icon as={ArrowRightIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
                    <Text flex="1" fontSize="md">
                      {t('widgets.layers.modifiers.moveToLayer')}
                    </Text>
                    <Icon as={ChevronRightIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
                  </HStack>
                </Menu.TriggerItem>
                <Portal>
                  <Menu.Positioner>
                    <MenuContent maxW="18rem" minW="10rem" py="1">
                      {moveTargets.map((target) => (
                        <MoveToLayerItem key={target.id} child={child} commands={commands} target={target} />
                      ))}
                    </MenuContent>
                  </Menu.Positioner>
                </Portal>
              </Menu.Root>
            )}
            <ChildMenuItem
              disabled={editingLocked}
              icon={XIcon}
              label={t(layerChildRemoveLabelKey(child.kind))}
              tone="danger"
              value="remove"
              onSelect={handleRemove}
            />
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const MoveToLayerItem = ({
  child,
  commands,
  target,
}: {
  child: ProjectedChildRow;
  commands: LayerRowCommands;
  target: { id: string; name: string };
}) => {
  const handleSelect = useCallback(() => commands.moveChildToLayer(child, target.id), [child, commands, target.id]);
  return <ChildMenuItem label={target.name} value={`move-to:${target.id}`} onSelect={handleSelect} />;
};
