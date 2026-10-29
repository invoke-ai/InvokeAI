/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelConfig } from '@features/models/core/types';
import type { ReactNode } from 'react';

import { Icon, Menu } from '@chakra-ui/react';
import { isConvertibleToDiffusers } from '@features/models/core/baseIdentity';
import { useModelActions } from '@features/models/ui/detail/useModelActions';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { HuggingFaceIcon } from '@platform/ui/VendoredIcon';
import { RefreshCcwIcon, Trash2Icon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/**
 * Keep confirmation dialogs outside menus so they survive menu closure; capture targets so disappearing source
 * rows cannot change pending actions.
 */

export type ModelActionsModel = Pick<ModelConfig, 'base' | 'format' | 'key' | 'name' | 'type'>;

export type PendingModelAction = { kind: 'convert' | 'delete' | 'reidentify'; model: ModelActionsModel } | null;

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
        <Menu.ItemText fontSize="md">{t('models.reidentify')}</Menu.ItemText>
      </Menu.Item>
      {showConvertItem && isConvertibleToDiffusers(model) ? (
        <Menu.Item value="convert" onClick={() => onRequestConfirm({ kind: 'convert', model })}>
          <Icon as={HuggingFaceIcon} boxSize="3.5" />
          <Menu.ItemText fontSize="md">{t('models.convertToDiffusers')}</Menu.ItemText>
        </Menu.Item>
      ) : null}
      {extraItems}
      <Menu.Separator />
      <Menu.Item data-danger="" value="delete" onClick={() => onRequestConfirm({ kind: 'delete', model })}>
        <Icon as={Trash2Icon} boxSize="3.5" />
        <Menu.ItemText fontSize="md">{t('models.deleteModel')}</Menu.ItemText>
      </Menu.Item>
    </>
  );
};

/**
 * Confirms whichever destructive action is pending: convert replaces the original checkpoint file; re-identify resets
 * the model's custom variant and default settings.
 */
const CONFIRM_COPY = {
  convert: { bodyKey: 'models.convertBody', confirmKey: 'models.convert', titleKey: 'models.convertToDiffusers' },
  delete: { bodyKey: 'models.deleteBody', confirmKey: 'models.deleteModel', titleKey: 'models.deleteModel' },
  reidentify: {
    bodyKey: 'models.reidentifyConfirmBody',
    confirmKey: 'models.reidentify',
    titleKey: 'models.reidentifyConfirmTitle',
  },
} as const;

export const ModelActionConfirmDialog = ({
  finalFocusEl,
  onClose,
  onDeleted,
  pending,
}: {
  /** Focus destination on close; list rows resolve it late because a delete removes the opener. */
  finalFocusEl?: () => HTMLElement | null;
  onClose: () => void;
  /** Called only when the delete completed in the current account scope. */
  onDeleted?: () => void;
  pending: PendingModelAction;
}) => {
  const { t } = useTranslation();
  const { convert, reidentify, remove } = useModelActions();
  const copy = CONFIRM_COPY[pending?.kind ?? 'delete'];

  return (
    <ConfirmDialog
      body={t(copy.bodyKey, { name: pending?.model.name ?? '' })}
      confirmLabel={t(copy.confirmKey)}
      finalFocusEl={finalFocusEl}
      isOpen={pending !== null}
      title={t(copy.titleKey)}
      onClose={onClose}
      onConfirm={async () => {
        if (!pending) {
          return;
        }

        if (pending.kind === 'convert') {
          await convert(pending.model);
          return;
        }

        if (pending.kind === 'reidentify') {
          await reidentify(pending.model);
          return;
        }

        if (await remove(pending.model)) {
          onDeleted?.();
        }
      }}
    />
  );
};
