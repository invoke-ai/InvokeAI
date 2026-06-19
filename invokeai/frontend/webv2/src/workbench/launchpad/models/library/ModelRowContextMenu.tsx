import type { ModelConfig } from '@workbench/models/types';

import { Icon, Menu, Portal } from '@chakra-ui/react';
import { ConfirmDialog, MenuContent } from '@workbench/components/ui';
import { useModelActions } from '@workbench/launchpad/models/detail/useModelActions';
import { isConvertibleToDiffusers } from '@workbench/models/baseIdentity';
import { RefreshCcwIcon, Trash2Icon } from 'lucide-react';
import { useRef, useState } from 'react';
import { SiHuggingface } from 'react-icons/si';

export interface ModelContextMenuTarget {
  model: ModelConfig;
  x: number;
  y: number;
}

/**
 * Right-click menu for library rows, anchored to the cursor via a virtual
 * rect. Mirrors the detail page's action menu (re-identify, convert, delete)
 * through the shared `useModelActions` hook.
 */
export const ModelRowContextMenu = ({
  onClose,
  target,
}: {
  onClose: () => void;
  target: ModelContextMenuTarget | null;
}) => {
  const { convert, reidentify, remove } = useModelActions();
  const [pendingConfirm, setPendingConfirm] = useState<{ kind: 'delete' | 'convert'; model: ModelConfig } | null>(null);
  const targetRef = useRef(target);

  targetRef.current = target;

  const model = target?.model ?? null;

  return (
    <>
      <Menu.Root
        key={target ? target.model.key : 'closed'}
        lazyMount
        open={target !== null}
        positioning={{
          getAnchorRect: () => {
            const currentTarget = targetRef.current;

            return currentTarget ? { height: 1, width: 1, x: currentTarget.x, y: currentTarget.y } : null;
          },
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
            {model ? (
              <MenuContent minW="13rem">
                <Menu.Item value="reidentify" onClick={() => void reidentify(model)}>
                  <Icon as={RefreshCcwIcon} boxSize="3.5" />
                  <Menu.ItemText fontSize="xs">Re-identify model</Menu.ItemText>
                </Menu.Item>
                {isConvertibleToDiffusers(model) ? (
                  <Menu.Item value="convert" onClick={() => setPendingConfirm({ kind: 'convert', model })}>
                    <Icon as={SiHuggingface} boxSize="3.5" />
                    <Menu.ItemText fontSize="xs">Convert to diffusers</Menu.ItemText>
                  </Menu.Item>
                ) : null}
                <Menu.Separator />
                <Menu.Item color="fg.error" value="delete" onClick={() => setPendingConfirm({ kind: 'delete', model })}>
                  <Icon as={Trash2Icon} boxSize="3.5" />
                  <Menu.ItemText fontSize="xs">Delete model</Menu.ItemText>
                </Menu.Item>
              </MenuContent>
            ) : null}
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <ConfirmDialog
        body={
          pendingConfirm?.kind === 'convert'
            ? `Convert “${pendingConfirm.model.name}” to the diffusers format in place? The original checkpoint file is replaced by a diffusers folder.`
            : `Delete “${pendingConfirm?.model.name}”? The database record is removed, and the model files are deleted if they live inside the InvokeAI models directory.`
        }
        confirmLabel={pendingConfirm?.kind === 'convert' ? 'Convert' : 'Delete Model'}
        isOpen={pendingConfirm !== null}
        title={pendingConfirm?.kind === 'convert' ? 'Convert to diffusers' : 'Delete model'}
        onClose={() => setPendingConfirm(null)}
        onConfirm={async () => {
          if (!pendingConfirm) {
            return;
          }

          await (pendingConfirm.kind === 'convert' ? convert(pendingConfirm.model) : remove(pendingConfirm.model));
        }}
      />
    </>
  );
};
