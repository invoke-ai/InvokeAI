import { Icon, Menu } from '@chakra-ui/react';
import { useCapabilities } from '@workbench/auth/capabilities';
import { getApiErrorMessage } from '@workbench/backend/http';
import { useNotify } from '@workbench/useNotify';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { ListOrderedIcon, PauseIcon, PlayIcon, XIcon, type LucideIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';

import { requestQueueConfirmation } from './queueConfirmationStore';
import { refreshQueue, useNowNextItems, useQueueCounts } from './queueDataStore';
import { useQueueQueryScope } from './queueScope';
import { cancelQueueItem, cancelScopedQueueItems, pauseQueueProcessor, resumeQueueProcessor } from './queueServerApi';

const ERROR_ITEM_HOVER_PROPS = { bg: 'bg.error', color: 'fg.error' };

interface QueueMenuActionInputs {
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
    label: 'Cancel Current Item',
    onClick: onCancelCurrent,
  },
  {
    destructive: true,
    disabled: cancellableCount === 0,
    icon: XIcon,
    label: 'Cancel All Items',
    onClick: onCancelAll,
  },
  {
    destructive: true,
    disabled: !isConnected || !hasPendingQueueWork,
    icon: XIcon,
    label: 'Cancel all except current item',
    onClick: onCancelAllExceptCurrent,
  },
  {
    disabled: !isConnected || !canManageProcessor,
    icon: PlayIcon,
    label: 'Resume Processor',
    onClick: onResumeProcessor,
  },
  {
    disabled: !isConnected || !canManageProcessor,
    icon: PauseIcon,
    label: 'Pause Processor',
    onClick: onPauseProcessor,
  },
  {
    disabled: false,
    icon: ListOrderedIcon,
    label: 'Open Queue',
    onClick: onOpenQueue,
  },
];

export const useQueueMenuActions = (): QueueMenuAction[] => {
  const backendConnectionStatus = useWorkbenchSelector((snapshot) => snapshot.state.backendConnection.status);
  const counts = useQueueCounts();
  const { current } = useNowNextItems();
  const scope = useQueueQueryScope();
  const { canManageModels } = useCapabilities();
  const notify = useNotify();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const isConnected = backendConnectionStatus === 'connected';
  const cancellableCount = counts.pending + counts.in_progress;
  const hasRunningItem = current?.status === 'in_progress';
  const hasPendingQueueWork = counts.pending > 0;

  const cancelCurrent = useCallback(() => {
    if (!current) {
      notify.info('No current item', 'The scoped queue has no in-progress item to cancel.');
      return;
    }

    cancelQueueItem(current.item_id)
      .then(() => refreshQueue())
      .then(() => notify.success('Cancellation requested', `Backend queue item ${current.item_id}.`))
      .catch((error: unknown) =>
        notify.error('Failed to cancel current item', getApiErrorMessage(error, 'Could not cancel the current item.'))
      );
  }, [current, notify]);

  const cancelAllExceptCurrent = useCallback(() => {
    cancelScopedQueueItems(scope, current?.item_id ?? null)
      .then(() => refreshQueue())
      .then(() =>
        notify.success(
          'Queue cancellation requested',
          'All pending scoped items except the current item were requested.'
        )
      )
      .catch((error: unknown) =>
        notify.error('Failed to cancel queue items', getApiErrorMessage(error, 'Could not cancel scoped queue items.'))
      );
  }, [current?.item_id, notify, scope]);

  const runProcessorAction = useCallback(
    (label: 'pause' | 'resume') => {
      const action = label === 'pause' ? pauseQueueProcessor : resumeQueueProcessor;

      action()
        .then(() => notify.success(label === 'pause' ? 'Processor paused' : 'Processor resumed'))
        .then(() => refreshQueue())
        .catch((error: unknown) =>
          notify.error(
            label === 'pause' ? 'Failed to pause processor' : 'Failed to resume processor',
            getApiErrorMessage(error, 'Could not change the queue processor.')
          )
        );
    },
    [notify]
  );
  const cancelAll = useCallback(() => {
    cancelScopedQueueItems(scope)
      .then(() => refreshQueue())
      .then(() => notify.success('Queue cancellation requested', 'All scoped queue items were requested.'))
      .catch((error: unknown) =>
        notify.error('Failed to cancel queue items', getApiErrorMessage(error, 'Could not cancel scoped queue items.'))
      );
  }, [notify, scope]);
  const resumeProcessor = useCallback(() => runProcessorAction('resume'), [runProcessorAction]);
  const pauseProcessor = useCallback(() => runProcessorAction('pause'), [runProcessorAction]);
  const openQueue = useCallback(() => openWorkbenchWidget('queue'), [openWorkbenchWidget]);

  return useMemo(
    () =>
      getQueueMenuActions({
        cancellableCount,
        canManageProcessor: canManageModels,
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
      canManageModels,
      hasPendingQueueWork,
      hasRunningItem,
      isConnected,
      openQueue,
      pauseProcessor,
      resumeProcessor,
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
  const onClick = useCallback(
    () =>
      requestQueueConfirmation({
        body: 'This requests cancellation for queue work in the current scope. In-progress items may stop after the backend acknowledges cancellation.',
        confirmLabel: action.label,
        onConfirm: action.onClick,
        title: `${action.label}?`,
      }),
    [action]
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
