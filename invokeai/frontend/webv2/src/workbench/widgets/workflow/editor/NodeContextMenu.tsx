import type { XYPosition } from '@workbench/workflows/types';

import { Icon, Menu, Portal } from '@chakra-ui/react';
import {
  ChevronsDownUpIcon,
  ChevronsUpDownIcon,
  ClipboardPasteIcon,
  CopyIcon,
  CopyPlusIcon,
  PlusIcon,
  Trash2Icon,
} from 'lucide-react';

/**
 * Right-click utilities for the flow pane and nodes, anchored at the pointer.
 * The menu is fully controlled because each action needs editor-local context.
 */

export interface NodeContextMenuState {
  kind: 'node';
  nodeId?: string;
  /** The node's `isOpen`; null for node types without a collapse toggle (notes). */
  isNodeOpen: boolean | null;
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
}) => (
  <Menu.Root
    open={menuState !== null}
    positioning={{
      getAnchorRect: () => (menuState ? { height: 1, width: 1, x: menuState.x, y: menuState.y } : null),
      placement: 'bottom-start',
    }}
    onOpenChange={(event) => {
      if (!event.open) {
        onClose();
      }
    }}
  >
    <Portal>
      <Menu.Positioner>
        <Menu.Content minW="11rem">
          {!menuState ? null : menuState.kind === 'pane' ? (
            <Menu.Item value="add-connector" onClick={() => onAddConnector(menuState.position)}>
              <Icon as={PlusIcon} boxSize="3.5" />
              <Menu.ItemText>Add connector</Menu.ItemText>
            </Menu.Item>
          ) : (
            <>
              <Menu.Item value="copy" onClick={onCopy}>
                <Icon as={CopyIcon} boxSize="3.5" />
                <Menu.ItemText>Copy</Menu.ItemText>
                <Menu.ItemCommand>Ctrl C</Menu.ItemCommand>
              </Menu.Item>
              <Menu.Item disabled={!canPaste} value="paste" _disabled={{ opacity: 0.4 }} onClick={onPaste}>
                <Icon as={ClipboardPasteIcon} boxSize="3.5" />
                <Menu.ItemText>Paste</Menu.ItemText>
                <Menu.ItemCommand>Ctrl V</Menu.ItemCommand>
              </Menu.Item>
              <Menu.Item value="duplicate" onClick={onDuplicate}>
                <Icon as={CopyPlusIcon} boxSize="3.5" />
                <Menu.ItemText>Duplicate</Menu.ItemText>
              </Menu.Item>
              {menuState.isNodeOpen !== null ? (
                <Menu.Item value="toggle-open" onClick={onToggleOpen}>
                  <Icon as={menuState.isNodeOpen ? ChevronsDownUpIcon : ChevronsUpDownIcon} boxSize="3.5" />
                  <Menu.ItemText>{menuState.isNodeOpen ? 'Collapse' : 'Expand'}</Menu.ItemText>
                </Menu.Item>
              ) : null}
              <Menu.Separator borderColor="border.subtle" />
              <Menu.Item color="fg.error" value="delete" onClick={onDelete}>
                <Icon as={Trash2Icon} boxSize="3.5" />
                <Menu.ItemText>Delete</Menu.ItemText>
                <Menu.ItemCommand>Del</Menu.ItemCommand>
              </Menu.Item>
            </>
          )}
        </Menu.Content>
      </Menu.Positioner>
    </Portal>
  </Menu.Root>
);
