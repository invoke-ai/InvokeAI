/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelConfig } from '@features/models/core/types';
import type { ReactNode } from 'react';

import { Icon, Menu } from '@chakra-ui/react';
import { isConvertibleToDiffusers } from '@features/models/core/baseIdentity';
import { useModelActions } from '@features/models/ui/detail/useModelActions';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { RefreshCcwIcon, Trash2Icon } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { SiHuggingface } from 'react-icons/si';

/**
 * The reidentify/convert/delete action set for one model, shared by the
 * library row context menu and the detail header menu so copy, icons, and
 * confirm semantics cannot drift. Items and dialog are separate components
 * because the dialog must outlive the menu (menus unmount their content on
 * close); the caller holds the pending state that bridges them, with the
 * target model captured so the confirm survives the source row disappearing.
 */

export type ModelActionsModel = Pick<ModelConfig, 'base' | 'format' | 'key' | 'name' | 'type'>;

export type PendingModelAction = { kind: 'convert' | 'delete'; model: ModelActionsModel } | null;

export const ModelActionMenuItems = ({
  extraItems,
  model,
  onBusyChange,
  onRequestConfirm,
  showConvertItem = false,
}: {
  /** Rendered between the standard items and the delete item. */
  extraItems?: ReactNode;
  model: ModelActionsModel;
  /** Reports the in-menu reidentify running, for a trigger spinner. */
  onBusyChange?: (isBusy: boolean) => void;
  onRequestConfirm: (pending: NonNullable<PendingModelAction>) => void;
  /** The detail header shows convert as its own button instead. */
  showConvertItem?: boolean;
}) => {
  const { t } = useTranslation();
  const { reidentify } = useModelActions();

  const handleReidentify = async () => {
    onBusyChange?.(true);

    try {
      await reidentify(model);
    } finally {
      onBusyChange?.(false);
    }
  };

  return (
    <>
      <Menu.Item value="reidentify" onClick={() => void handleReidentify()}>
        <Icon as={RefreshCcwIcon} boxSize="3.5" />
        <Menu.ItemText fontSize="xs">{t('models.reidentify')}</Menu.ItemText>
      </Menu.Item>
      {showConvertItem && isConvertibleToDiffusers(model) ? (
        <Menu.Item value="convert" onClick={() => onRequestConfirm({ kind: 'convert', model })}>
          <Icon as={SiHuggingface} boxSize="3.5" />
          <Menu.ItemText fontSize="xs">{t('models.convertToDiffusers')}</Menu.ItemText>
        </Menu.Item>
      ) : null}
      {extraItems}
      <Menu.Separator />
      <Menu.Item color="fg.error" value="delete" onClick={() => onRequestConfirm({ kind: 'delete', model })}>
        <Icon as={Trash2Icon} boxSize="3.5" />
        <Menu.ItemText fontSize="xs">{t('models.deleteModel')}</Menu.ItemText>
      </Menu.Item>
    </>
  );
};

/** Confirms whichever destructive action is pending (convert replaces the original checkpoint file). */
export const ModelActionConfirmDialog = ({
  onClose,
  onDeleted,
  pending,
}: {
  onClose: () => void;
  /** Called only when the delete completed in the current account scope. */
  onDeleted?: () => void;
  pending: PendingModelAction;
}) => {
  const { t } = useTranslation();
  const { convert, remove } = useModelActions();

  return (
    <ConfirmDialog
      body={
        pending?.kind === 'convert'
          ? t('models.convertBody', { name: pending.model.name })
          : t('models.deleteBody', { name: pending?.model.name ?? '' })
      }
      confirmLabel={pending?.kind === 'convert' ? t('models.convert') : t('models.deleteModel')}
      isOpen={pending !== null}
      title={pending?.kind === 'convert' ? t('models.convertToDiffusers') : t('models.deleteModel')}
      onClose={onClose}
      onConfirm={async () => {
        if (!pending) {
          return;
        }

        if (pending.kind === 'convert') {
          await convert(pending.model);
          return;
        }

        if (await remove(pending.model)) {
          onDeleted?.();
        }
      }}
    />
  );
};
