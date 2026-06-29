/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { NodePackInfo } from '@workbench/customNodes/api';

import { Icon, Menu, Portal } from '@chakra-ui/react';
import { ConfirmDialog, MenuContent } from '@workbench/components/ui';
import { useNodePackActions } from '@workbench/launchpad/nodes/shared/useNodePackActions';
import { Trash2Icon } from 'lucide-react';
import { useState } from 'react';

export interface NodePackContextMenuTarget {
  pack: NodePackInfo;
  x: number;
  y: number;
}

export const NodePackContextMenu = ({
  onClose,
  onUninstalled,
  target,
}: {
  onClose: () => void;
  onUninstalled: (packName: string) => void;
  target: NodePackContextMenuTarget | null;
}) => {
  const { uninstall } = useNodePackActions();
  const [pendingUninstall, setPendingUninstall] = useState<NodePackInfo | null>(null);
  const pack = target?.pack ?? null;

  return (
    <>
      <Menu.Root
        key={target ? target.pack.name : 'closed'}
        lazyMount
        open={target !== null}
        positioning={{
          getAnchorRect: () => (target ? { height: 1, width: 1, x: target.x, y: target.y } : null),
          placement: 'bottom-start',
        }}
        unmountOnExit
        onOpenChange={(event) => {
          if (!event.open) {
            onClose();
          }
        }}
      >
        <Portal>
          <Menu.Positioner>
            {pack ? (
              <MenuContent minW="12rem">
                <Menu.Item color="fg.error" value="uninstall" onClick={() => setPendingUninstall(pack)}>
                  <Icon as={Trash2Icon} boxSize="3.5" />
                  <Menu.ItemText fontSize="xs">Uninstall</Menu.ItemText>
                </Menu.Item>
              </MenuContent>
            ) : null}
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <ConfirmDialog
        body="Remove this pack from the custom nodes directory? A restart is required for removal to fully apply."
        confirmLabel="Uninstall Node Pack"
        isOpen={pendingUninstall !== null}
        title={`Uninstall ${pendingUninstall?.name ?? 'node pack'}?`}
        onClose={() => setPendingUninstall(null)}
        onConfirm={async () => {
          if (!pendingUninstall) {
            return;
          }

          await uninstall(pendingUninstall, onUninstalled);
        }}
      />
    </>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
