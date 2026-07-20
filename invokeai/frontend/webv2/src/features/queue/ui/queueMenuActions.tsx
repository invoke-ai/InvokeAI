import { Icon, Menu } from '@chakra-ui/react';
import { queueCommands } from '@features/queue/publicApi';
import { getApiErrorMessage } from '@platform/transport/http';
import { ListOrderedIcon, PauseIcon, PlayIcon, XIcon, type LucideIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { requestQueueConfirmation } from './queueConfirmationStore';
import { refreshQueue, useNowNextItems, useQueueCounts } from './queueDataStore';
import { useQueueQueryScope } from './queueScope';
import { useQueueUi } from './QueueUiContext';

const ERROR_ITEM_HOVER_PROPS = { bg: 'bg.error', color: 'fg.error' };

interface QueueMenuActionInputs {
  labels: {
    cancelAll: string;
    cancelAllExceptCurrent: string;
    cancelCurrent: string;
    openQueue: string;
    pauseProcessor: string;
    resumeProcessor: string;
  };
  cancellableCount: number;
  hasPendingQueueWork: boolean;
  hasRunningItem: boolean;
  isConnected: boolean;
  canManageProcessor: boolean;
  onCancelAll: () => void;
  onCancelAllExceptCurrent: () => void;
  onCancelCurrent: () => void;
  onOpenQueue: () => void;
  onPauseProcessor: () => void;
  onResumeProcessor: () => void;
}

export interface QueueMenuAction {
  label: string;
  icon: LucideIcon;
  disabled: boolean;
  destructive?: boolean;
  onClick: () => void;
}

export const getQueueMenuActions = ({
  labels,
  cancellableCount,
  hasPendingQueueWork,
  hasRunningItem,
  isConnected,
  canManageProcessor,
  onCancelAll,
  onCancelAllExceptCurrent,
  onCancelCurrent,
  onOpenQueue,
  onPauseProcessor,
  onResumeProcessor,
}: QueueMenuActionInputs): QueueMenuAction[] => [
  {
    destructive: true,
    disabled: !isConnected || !hasRunningItem,
    icon: XIcon,
    label: labels.cancelCurrent,
    onClick: onCancelCurrent,
  },
  {
    destructive: true,
    disabled: cancellableCount === 0,
    icon: XIcon,
    label: labels.cancelAll,
    onClick: onCancelAll,
  },
  {
    destructive: true,
    disabled: !isConnected || !hasPendingQueueWork,
    icon: XIcon,
    label: labels.cancelAllExceptCurrent,
    onClick: onCancelAllExceptCurrent,
  },
  {
    disabled: !isConnected || !canManageProcessor,
    icon: PlayIcon,
    label: labels.resumeProcessor,
    onClick: onResumeProcessor,
  },
  {
    disabled: !isConnected || !canManageProcessor,
    icon: PauseIcon,
    label: labels.pauseProcessor,
    onClick: onPauseProcessor,
  },
  {
    disabled: false,
    icon: ListOrderedIcon,
    label: labels.openQueue,
    onClick: onOpenQueue,
  },
];

export const useQueueMenuActions = (): QueueMenuAction[] => {
  const { t } = useTranslation();
  const counts = useQueueCounts();
  const { current } = useNowNextItems();
  const scope = useQueueQueryScope();
  const { canManageProcessor, isConnected, notify, openQueue } = useQueueUi();
  const cancellableCount = counts.pending + counts.inProgress;
  const hasRunningItem = current?.status === 'in_progress';
  const hasPendingQueueWork = counts.pending > 0;

  const cancelCurrent = useCallback(() => {
    if (!current) {
      notify.info(t('widgets.queue.noCurrentItem'), t('widgets.queue.noCurrentItemDescription'));
      return;
    }

    queueCommands
      .cancelItem(current.id)
      .then(() => refreshQueue())
      .then(() =>
        notify.success(t('widgets.queue.cancellationRequested'), t('widgets.queue.backendItem', { id: current.id }))
      )
      .catch((error: unknown) =>
        notify.error(
          t('widgets.queue.failedToCancelCurrentItem'),
          getApiErrorMessage(error, t('widgets.queue.couldNotCancelCurrentItem'))
        )
      );
  }, [current, notify, t]);

  const cancelAllExceptCurrent = useCallback(() => {
    queueCommands
      .cancelScopedItems(scope, current?.id ?? null)
      .then(() => refreshQueue())
      .then(() =>
        notify.success(
          t('widgets.queue.queueCancellationRequested'),
          t('widgets.queue.cancelAllExceptCurrentRequested')
        )
      )
      .catch((error: unknown) =>
        notify.error(
          t('widgets.queue.failedToCancelItems'),
          getApiErrorMessage(error, t('widgets.queue.couldNotCancelItems'))
        )
      );
  }, [current?.id, notify, scope, t]);

  const runProcessorAction = useCallback(
    (label: 'pause' | 'resume') => {
      const action = label === 'pause' ? queueCommands.pauseProcessor : queueCommands.resumeProcessor;

      action()
        .then(() =>
          notify.success(label === 'pause' ? t('widgets.queue.processorPaused') : t('widgets.queue.processorResumed'))
        )
        .then(() => refreshQueue())
        .catch((error: unknown) =>
          notify.error(
            label === 'pause' ? t('widgets.queue.failedToPauseProcessor') : t('widgets.queue.failedToResumeProcessor'),
            getApiErrorMessage(error, t('widgets.queue.couldNotChangeProcessor'))
          )
        );
    },
    [notify, t]
  );
  const cancelAll = useCallback(() => {
    queueCommands
      .cancelScopedItems(scope)
      .then(() => refreshQueue())
      .then(() =>
        notify.success(t('widgets.queue.queueCancellationRequested'), t('widgets.queue.allScopedItemsRequested'))
      )
      .catch((error: unknown) =>
        notify.error(
          t('widgets.queue.failedToCancelItems'),
          getApiErrorMessage(error, t('widgets.queue.couldNotCancelItems'))
        )
      );
  }, [notify, scope, t]);
  const resumeProcessor = useCallback(() => runProcessorAction('resume'), [runProcessorAction]);
  const pauseProcessor = useCallback(() => runProcessorAction('pause'), [runProcessorAction]);

  return useMemo(
    () =>
      getQueueMenuActions({
        labels: {
          cancelAll: t('widgets.queue.cancelAllItems'),
          cancelAllExceptCurrent: t('widgets.queue.cancelAllExceptCurrent'),
          cancelCurrent: t('widgets.queue.cancelCurrentItem'),
          openQueue: t('widgets.queue.openQueue'),
          pauseProcessor: t('widgets.queue.pauseProcessor'),
          resumeProcessor: t('widgets.queue.resumeProcessor'),
        },
        cancellableCount,
        canManageProcessor,
        hasPendingQueueWork,
        hasRunningItem,
        isConnected,
        onCancelAll: cancelAll,
        onCancelAllExceptCurrent: cancelAllExceptCurrent,
        onCancelCurrent: cancelCurrent,
        onOpenQueue: openQueue,
        onPauseProcessor: pauseProcessor,
        onResumeProcessor: resumeProcessor,
      }),
    [
      cancelAll,
      cancelAllExceptCurrent,
      cancelCurrent,
      cancellableCount,
      canManageProcessor,
      hasPendingQueueWork,
      hasRunningItem,
      isConnected,
      openQueue,
      pauseProcessor,
      resumeProcessor,
      t,
    ]
  );
};

export const QueueMenuItems = ({ actions, label }: { actions: QueueMenuAction[]; label?: string }) => {
  const destructiveActions = actions.filter((action) => action.destructive);
  const processorActions = actions.filter((action) => !action.destructive);

  return (
    <>
      <Menu.ItemGroup>
        {label ? (
          <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
            {label}
          </Menu.ItemGroupLabel>
        ) : null}
        {destructiveActions.map((action) => (
          <DestructiveQueueMenuItem key={action.label} action={action} />
        ))}
      </Menu.ItemGroup>
      <Menu.Separator />
      <Menu.ItemGroup>
        {processorActions.map((action) => (
          <Menu.Item key={action.label} disabled={action.disabled} value={action.label} onClick={action.onClick}>
            <Icon as={action.icon} boxSize="3" />
            <Menu.ItemText>{action.label}</Menu.ItemText>
          </Menu.Item>
        ))}
      </Menu.ItemGroup>
    </>
  );
};

const DestructiveQueueMenuItem = ({ action }: { action: QueueMenuAction }) => {
  const { t } = useTranslation();
  const onClick = useCallback(
    () =>
      requestQueueConfirmation({
        body: t('widgets.queue.cancelConfirmationBody'),
        confirmLabel: action.label,
        onConfirm: action.onClick,
        title: t('widgets.queue.actionConfirmationTitle', { action: action.label }),
      }),
    [action, t]
  );

  return (
    <Menu.Item
      color="fg.error"
      disabled={action.disabled}
      value={action.label}
      _hover={ERROR_ITEM_HOVER_PROPS}
      onClick={onClick}
    >
      <Icon as={action.icon} boxSize="3" />
      <Menu.ItemText>{action.label}</Menu.ItemText>
    </Menu.Item>
  );
};
