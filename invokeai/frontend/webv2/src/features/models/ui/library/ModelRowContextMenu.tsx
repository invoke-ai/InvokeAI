/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ListContextMenuAnchor } from '@platform/ui/list/ListItem';

import { Menu, Portal } from '@chakra-ui/react';
import { useModelsSelector } from '@features/models/data/modelsStore';
import {
  ModelActionConfirmDialog,
  ModelActionMenuItems,
  type PendingModelAction,
} from '@features/models/ui/shared/ModelActionsMenu';
import { useExitRetainedValue } from '@platform/react/useExitRetainedValue';
import { MenuContent } from '@platform/ui';
import { useState } from 'react';

export interface ModelContextMenuTarget extends ListContextMenuAnchor {
  modelKey: string;
}

export const ModelRowContextMenu = ({
  onClose,
  target,
}: {
  onClose: () => void;
  target: ModelContextMenuTarget | null;
}) => {
  const [pendingConfirm, setPendingConfirm] = useState<PendingModelAction>(null);
  // Captured with the request: the menu target clears when the menu closes, before the dialog does.
  const [confirmFocusTarget, setConfirmFocusTarget] = useState<(() => HTMLElement | null) | null>(null);
  // Kept through the exit animation; keyed by model so another row's menu still opens at its own anchor.
  const { release, value: shown } = useExitRetainedValue(target);
  const model = useModelsSelector((snapshot) => (shown ? (snapshot.modelsByKey.get(shown.modelKey) ?? null) : null));

  return (
    <>
      <Menu.Root
        key={shown ? shown.modelKey : 'none'}
        lazyMount
        open={target !== null}
        positioning={{
          getAnchorRect: () => (shown ? { height: 1, width: 1, x: shown.x, y: shown.y } : null),
          placement: 'bottom-start',
        }}
        unmountOnExit
        onExitComplete={release}
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
                <ModelActionMenuItems
                  model={model}
                  showConvertItem
                  onRequestConfirm={(pending) => {
                    setConfirmFocusTarget(() => target?.focusTarget ?? null);
                    setPendingConfirm(pending);
                  }}
                />
              </MenuContent>
            ) : null}
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <ModelActionConfirmDialog
        finalFocusEl={confirmFocusTarget ?? undefined}
        pending={pendingConfirm}
        onClose={() => setPendingConfirm(null)}
      />
    </>
  );
};
