import { Icon, Menu } from '@chakra-ui/react';
import { queueCommands } from '@features/queue/publicApi';
import { getApiErrorMessage } from '@platform/transport/http';
import { Trash2Icon, TrashIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { requestQueueConfirmation } from './queueConfirmationStore';
import { refreshQueue, useQueueCounts } from './queueDataStore';
import { QueueMenuItems, useQueueMenuActions } from './queueMenuActions';
import { useQueueQueryScope } from './queueScope';
import { useQueueUi } from './QueueUiContext';

const ERROR_ITEM_HOVER_PROPS = { bg: 'bg.error', color: 'fg.error' };

/**
 * Queue actions contributed to the frame's shared actions menu via the manifest
 * `headerMenu` slot. This intentionally uses the same action model as the
 * topbar queue menu so backend mutations keep local run coordination in sync.
 */
export const QueueHeaderMenu = () => {
  const { t } = useTranslation();
  const actions = useQueueMenuActions();
  const counts = useQueueCounts();
  const scope = useQueueQueryScope();
  const { notify } = useQueueUi();

  const clearFailed = useCallback(async () => {
    try {
      await queueCommands.clearFailedItems(scope);
      await refreshQueue();
      notify.success(t('widgets.queue.failedItemsCleared'));
    } catch (error) {
      notify.error(
        t('widgets.queue.failedToClearItems'),
        getApiErrorMessage(error, t('widgets.queue.couldNotClearFailedItems'))
      );
    }
  }, [notify, scope, t]);

  const clearAll = useCallback(async () => {
    try {
      await queueCommands.clearItems(scope);
      await refreshQueue();
      notify.success(t('widgets.queue.queueCleared'));
    } catch (error) {
      notify.error(
        t('widgets.queue.failedToClearQueue'),
        getApiErrorMessage(error, t('widgets.queue.couldNotClearQueue'))
      );
    }
  }, [notify, scope, t]);
  const onClearFailed = useCallback(() => {
    requestQueueConfirmation({
      body: t('widgets.queue.clearFailedConfirmationBody'),
      confirmLabel: t('widgets.queue.clearFailedItems'),
      onConfirm: clearFailed,
      title: t('widgets.queue.clearFailedTitle'),
    });
  }, [clearFailed, t]);
  const onClearAll = useCallback(() => {
    requestQueueConfirmation({
      body: t('widgets.queue.clearQueueConfirmationBody'),
      confirmLabel: t('widgets.queue.clearQueue'),
      onConfirm: clearAll,
      title: t('widgets.queue.clearQueueTitle'),
    });
  }, [clearAll, t]);

  return (
    <>
      <QueueMenuItems actions={actions} label={t('widgets.labels.queue')} />
      <Menu.Separator />
      <Menu.ItemGroup>
        <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
          {t('common.clear')}
        </Menu.ItemGroupLabel>
        <Menu.Item
          color="fg.error"
          disabled={counts.failed === 0}
          value="clear-failed-items"
          _hover={ERROR_ITEM_HOVER_PROPS}
          onClick={onClearFailed}
        >
          <Icon as={Trash2Icon} boxSize="3" />
          <Menu.ItemText>{t('widgets.queue.clearFailedItems')}</Menu.ItemText>
        </Menu.Item>
        <Menu.Item
          color="fg.error"
          disabled={counts.total === 0}
          value="clear-all-items"
          _hover={ERROR_ITEM_HOVER_PROPS}
          onClick={onClearAll}
        >
          <Icon as={TrashIcon} boxSize="3" />
          <Menu.ItemText>{t('widgets.queue.clearAllItems')}</Menu.ItemText>
        </Menu.Item>
      </Menu.ItemGroup>
    </>
  );
};
