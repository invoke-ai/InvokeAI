/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { NodePackInfo } from '@features/nodes/core/catalog';
import type { ListContextMenuAnchor } from '@platform/ui/list/ListItem';

import { Icon, Menu, Portal } from '@chakra-ui/react';
import { UninstallPackDialog } from '@features/nodes/ui/shared/UninstallPackDialog';
import { useNotify } from '@features/nodes/ui/useNodesNotify';
import { MenuContent } from '@platform/ui';
import { ClipboardCopyIcon, Trash2Icon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

export interface NodePackContextMenuTarget extends ListContextMenuAnchor {
  pack: NodePackInfo;
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
  const { t } = useTranslation();
  const notify = useNotify();
  const [pendingUninstall, setPendingUninstall] = useState<NodePackInfo | null>(null);
  // Captured with the request: the menu target clears when the menu closes, before the dialog does.
  const [uninstallFocusTarget, setUninstallFocusTarget] = useState<(() => HTMLElement | null) | null>(null);
  const pack = target?.pack ?? null;

  const handleCopyPath = async (path: string) => {
    try {
      await navigator.clipboard.writeText(path);
      notify.success(t('nodes.pathCopied'));
    } catch {
      notify.error(t('common.couldNotCopy'));
    }
  };

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
                <Menu.Item value="copy-path" onClick={() => void handleCopyPath(pack.path)}>
                  <Icon as={ClipboardCopyIcon} boxSize="3.5" />
                  <Menu.ItemText fontSize="md">{t('nodes.copyPath')}</Menu.ItemText>
                </Menu.Item>
                <Menu.Separator />
                <Menu.Item
                  data-danger=""
                  value="uninstall"
                  onClick={() => {
                    setUninstallFocusTarget(() => target?.focusTarget ?? null);
                    setPendingUninstall(pack);
                  }}
                >
                  <Icon as={Trash2Icon} boxSize="3.5" />
                  <Menu.ItemText fontSize="md">{t('nodes.uninstall')}</Menu.ItemText>
                </Menu.Item>
              </MenuContent>
            ) : null}
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <UninstallPackDialog
        finalFocusEl={uninstallFocusTarget ?? undefined}
        pack={pendingUninstall}
        onClose={() => setPendingUninstall(null)}
        onUninstalled={onUninstalled}
      />
    </>
  );
};
