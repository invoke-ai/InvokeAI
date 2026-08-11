/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Menu, Portal } from '@chakra-ui/react';
import { useModelsSelector } from '@features/models/data/modelsStore';
import {
  ModelActionConfirmDialog,
  ModelActionMenuItems,
  type PendingModelAction,
} from '@features/models/ui/shared/ModelActionsMenu';
import { MenuContent } from '@platform/ui';
import { useState } from 'react';

export interface ModelContextMenuTarget {
  modelKey: string;
  x: number;
  y: number;
}

/**
 * Right-click menu for library rows, anchored to the cursor via a virtual
 * rect. Mirrors the detail page's action menu (re-identify, convert, delete)
 * through the shared `ModelActionMenuItems`.
 */
export const ModelRowContextMenu = ({
  onClose,
  target,
}: {
  onClose: () => void;
  target: ModelContextMenuTarget | null;
}) => {
  const [pendingConfirm, setPendingConfirm] = useState<PendingModelAction>(null);
  const model = useModelsSelector((snapshot) => (target ? (snapshot.modelsByKey.get(target.modelKey) ?? null) : null));

  return (
    <>
      <Menu.Root
        key={target ? target.modelKey : 'closed'}
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
            {model ? (
              <MenuContent minW="13rem">
                <ModelActionMenuItems model={model} showConvertItem onRequestConfirm={setPendingConfirm} />
              </MenuContent>
            ) : null}
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <ModelActionConfirmDialog pending={pendingConfirm} onClose={() => setPendingConfirm(null)} />
    </>
  );
};
