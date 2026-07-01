/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelConfig } from '@workbench/models/types';

import { Icon, Menu, Portal } from '@chakra-ui/react';
import { ConfirmDialog, MenuContent } from '@workbench/components/ui';
import { useModelActions } from '@workbench/launchpad/models/detail/useModelActions';
import { isConvertibleToDiffusers } from '@workbench/models/baseIdentity';
import { useModelsSelector } from '@workbench/models/modelsStore';
import { RefreshCcwIcon, Trash2Icon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { SiHuggingface } from 'react-icons/si';

export interface ModelContextMenuTarget {
  modelKey: string;
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
  const { t } = useTranslation();
  const { convert, reidentify, remove } = useModelActions();
  const [pendingConfirm, setPendingConfirm] = useState<{ kind: 'delete' | 'convert'; model: ModelConfig } | null>(null);
  const model = useModelsSelector((snapshot) =>
    target ? (snapshot.models.find((candidate) => candidate.key === target.modelKey) ?? null) : null
  );

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
                <Menu.Item value="reidentify" onClick={() => void reidentify(model)}>
                  <Icon as={RefreshCcwIcon} boxSize="3.5" />
                  <Menu.ItemText fontSize="xs">{t('models.reidentify')}</Menu.ItemText>
                </Menu.Item>
                {isConvertibleToDiffusers(model) ? (
                  <Menu.Item value="convert" onClick={() => setPendingConfirm({ kind: 'convert', model })}>
                    <Icon as={SiHuggingface} boxSize="3.5" />
                    <Menu.ItemText fontSize="xs">{t('models.convertToDiffusers')}</Menu.ItemText>
                  </Menu.Item>
                ) : null}
                <Menu.Separator />
                <Menu.Item color="fg.error" value="delete" onClick={() => setPendingConfirm({ kind: 'delete', model })}>
                  <Icon as={Trash2Icon} boxSize="3.5" />
                  <Menu.ItemText fontSize="xs">{t('models.deleteModel')}</Menu.ItemText>
                </Menu.Item>
              </MenuContent>
            ) : null}
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <ConfirmDialog
        body={
          pendingConfirm?.kind === 'convert'
            ? t('models.convertBody', { name: pendingConfirm.model.name })
            : t('models.deleteBody', { name: pendingConfirm?.model.name ?? '' })
        }
        confirmLabel={pendingConfirm?.kind === 'convert' ? t('models.convert') : t('models.deleteModel')}
        isOpen={pendingConfirm !== null}
        title={pendingConfirm?.kind === 'convert' ? t('models.convertToDiffusers') : t('models.deleteModel')}
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
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
