import type { XYPosition } from '@features/workflow/contracts';

import { Icon, Menu, Portal } from '@chakra-ui/react';
import { MenuContent } from '@platform/ui/Menu';
import {
  ChevronsDownUpIcon,
  ChevronsUpDownIcon,
  CircleArrowUpIcon,
  ClipboardPasteIcon,
  CopyIcon,
  CopyPlusIcon,
  PlusIcon,
  Trash2Icon,
} from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export interface NodeContextMenuState {
  kind: 'node';
  nodeId?: string;
  /** The node's `isOpen`; null for node types without a collapse toggle (notes). */
  isNodeOpen: boolean | null;
  /** The node's template is a newer same-major version, so it can be updated in place. */
  canUpdate: boolean;
  x: number;
  y: number;
}

export interface PaneContextMenuState {
  kind: 'pane';
  position: XYPosition;
  x: number;
  y: number;
}

export type WorkflowContextMenuState = NodeContextMenuState | PaneContextMenuState;

export const NodeContextMenu = ({
  canPaste,
  menuState,
  onClose,
  onAddConnector,
  onCopy,
  onDelete,
  onDuplicate,
  onPaste,
  onToggleOpen,
  onUpdate,
}: {
  canPaste: boolean;
  menuState: WorkflowContextMenuState | null;
  onClose: () => void;
  onAddConnector: (position: XYPosition) => void;
  onCopy: () => void;
  onDelete: () => void;
  onDuplicate: () => void;
  onPaste: () => void;
  onToggleOpen: () => void;
  onUpdate: () => void;
}) => {
  const { t } = useTranslation();
  const positioning = useMemo(
    () => ({
      getAnchorRect: () => (menuState ? { height: 1, width: 1, x: menuState.x, y: menuState.y } : null),
      placement: 'bottom-start' as const,
    }),
    [menuState]
  );
  const onOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );

  return (
    <Menu.Root open={menuState !== null} positioning={positioning} onOpenChange={onOpenChange}>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="11rem">
            {!menuState ? null : menuState.kind === 'pane' ? (
              <PaneAddConnectorMenuItem position={menuState.position} onAddConnector={onAddConnector} />
            ) : (
              <>
                <Menu.Item value="copy" onClick={onCopy}>
                  <Icon as={CopyIcon} boxSize="3.5" />
                  <Menu.ItemText>{t('common.copy')}</Menu.ItemText>
                  <Menu.ItemCommand>Ctrl C</Menu.ItemCommand>
                </Menu.Item>
                <Menu.Item disabled={!canPaste} value="paste" _disabled={DISABLED_PROPS} onClick={onPaste}>
                  <Icon as={ClipboardPasteIcon} boxSize="3.5" />
                  <Menu.ItemText>{t('nodes.contextPaste')}</Menu.ItemText>
                  <Menu.ItemCommand>Ctrl V</Menu.ItemCommand>
                </Menu.Item>
                <Menu.Item value="duplicate" onClick={onDuplicate}>
                  <Icon as={CopyPlusIcon} boxSize="3.5" />
                  <Menu.ItemText>{t('common.duplicate')}</Menu.ItemText>
                </Menu.Item>
                {menuState.isNodeOpen !== null ? (
                  <Menu.Item value="toggle-open" onClick={onToggleOpen}>
                    <Icon as={menuState.isNodeOpen ? ChevronsDownUpIcon : ChevronsUpDownIcon} boxSize="3.5" />
                    <Menu.ItemText>
                      {menuState.isNodeOpen ? t('nodes.contextCollapse') : t('nodes.contextExpand')}
                    </Menu.ItemText>
                  </Menu.Item>
                ) : null}
                {menuState.canUpdate ? (
                  <Menu.Item value="update" onClick={onUpdate}>
                    <Icon as={CircleArrowUpIcon} boxSize="3.5" />
                    <Menu.ItemText>{t('nodes.updateNode')}</Menu.ItemText>
                  </Menu.Item>
                ) : null}
                <Menu.Separator borderColor="border.subtle" />
                <Menu.Item data-danger="" value="delete" onClick={onDelete}>
                  <Icon as={Trash2Icon} boxSize="3.5" />
                  <Menu.ItemText>{t('common.delete')}</Menu.ItemText>
                  <Menu.ItemCommand>Del</Menu.ItemCommand>
                </Menu.Item>
              </>
            )}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const DISABLED_PROPS = { opacity: 0.4 };

const PaneAddConnectorMenuItem = ({
  onAddConnector,
  position,
}: {
  onAddConnector: (position: XYPosition) => void;
  position: XYPosition;
}) => {
  const { t } = useTranslation();
  const onClick = useCallback(() => onAddConnector(position), [onAddConnector, position]);

  return (
    <Menu.Item value="add-connector" onClick={onClick}>
      <Icon as={PlusIcon} boxSize="3.5" />
      <Menu.ItemText>{t('nodes.contextAddConnector')}</Menu.ItemText>
    </Menu.Item>
  );
};
