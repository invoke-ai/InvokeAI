import { Group, Icon, Menu, Portal } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@workbench/components/ui';
import {
  cancelAllExceptCurrentQueueItems,
  cancelCurrentQueueItem,
  pauseQueueProcessor,
  resumeQueueProcessor,
} from '@workbench/generation/api';
import { useNotify } from '@workbench/useNotify';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { getQueueActionState } from '@workbench/widgets/queue/queueViewModel';
import { useActiveProjectSelector, useWorkbenchDispatch, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { ChevronDownIcon, ListOrderedIcon, PauseIcon, PlayIcon, XIcon } from 'lucide-react';

/** Queue cancel cluster and processor actions. */
export const QueueActions = () => {
  const activeProjectId = useWorkbenchSelector((snapshot) => snapshot.state.activeProjectId);
  const backendConnectionStatus = useWorkbenchSelector((snapshot) => snapshot.state.backendConnection.status);
  const { cancellableCount, currentItemId, hasPendingQueueWork, hasRunningItem } = useActiveProjectSelector((project) =>
    getQueueActionState(project.queue.items)
  );
  const dispatch = useWorkbenchDispatch();
  const notify = useNotify();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const isConnected = backendConnectionStatus === 'connected';

  const cancelCurrent = () => {
    cancelCurrentQueueItem()
      .then((queueItem) => {
        if (!queueItem) {
          notify.info('No current item', 'The backend queue has no in-progress item to cancel.');
          return;
        }

        notify.success('Cancellation requested', `Backend queue item ${queueItem.item_id}.`);
        dispatch({ type: 'refreshBackendData' });
      })
      .catch((error: unknown) =>
        notify.error('Failed to cancel current item', error instanceof Error ? error.message : String(error))
      );
  };

  const cancelAllExceptCurrent = () => {
    cancelAllExceptCurrentQueueItems()
      .then(() => {
        dispatch({
          currentQueueItemId: currentItemId,
          projectId: activeProjectId,
          type: 'cancelAllQueueItemsExceptCurrent',
        });
        dispatch({ type: 'refreshBackendData' });
        notify.success('Queue cancellation requested', 'All pending items except the current item were requested.');
      })
      .catch((error: unknown) =>
        notify.error('Failed to cancel queue items', error instanceof Error ? error.message : String(error))
      );
  };

  const runProcessorAction = (label: 'pause' | 'resume') => {
    const action = label === 'pause' ? pauseQueueProcessor : resumeQueueProcessor;

    action()
      .then(() => notify.success(label === 'pause' ? 'Processor paused' : 'Processor resumed'))
      .catch((error: unknown) =>
        notify.error(
          label === 'pause' ? 'Failed to pause processor' : 'Failed to resume processor',
          error instanceof Error ? error.message : String(error)
        )
      );
  };

  const queueEndActions = [
    {
      label: 'Cancel Current Item',
      icon: XIcon,
      disabled: !isConnected || !hasRunningItem,
      onClick: cancelCurrent,
    },
    {
      label: 'Cancel All Items',
      icon: XIcon,
      disabled: cancellableCount === 0,
      onClick: () => dispatch({ projectId: activeProjectId, type: 'cancelAllQueueItems' }),
    },
    {
      label: 'Cancel all except current item',
      icon: XIcon,
      disabled: !isConnected || !hasPendingQueueWork,
      onClick: cancelAllExceptCurrent,
    },
  ];

  const queueProcessorActions = [
    {
      label: 'Resume Processor',
      icon: PlayIcon,
      disabled: !isConnected,
      onClick: () => runProcessorAction('resume'),
    },
    {
      label: 'Pause Processor',
      icon: PauseIcon,
      disabled: !isConnected,
      onClick: () => runProcessorAction('pause'),
    },
    {
      label: 'Open Queue',
      icon: ListOrderedIcon,
      disabled: false,
      onClick: () => openWorkbenchWidget('queue'),
    },
  ];

  return (
    <Menu.Root>
      <Group attached>
        <Tooltip content="Cancel current item" showArrow>
          <IconButton
            aria-label="Cancel current queue item"
            disabled={!isConnected || !hasRunningItem}
            variant="outline"
            size="sm"
            roundedEnd="none"
            borderColor="border.subtle"
            onClick={cancelCurrent}
          >
            <XIcon />
          </IconButton>
        </Tooltip>
        <Menu.Trigger asChild>
          <IconButton
            aria-label="Queue actions"
            variant="outline"
            size="sm"
            roundedStart="none"
            borderStartWidth="0"
            aspectRatio="unset"
            minW="0"
            w="6"
            borderColor="border.subtle"
          >
            <ChevronDownIcon />
          </IconButton>
        </Menu.Trigger>
      </Group>
      <Portal>
        <Menu.Positioner>
          <Menu.Content>
            <Menu.ItemGroup>
              {queueEndActions.map((action, index) => (
                <Menu.Item
                  key={index}
                  onClick={action.onClick}
                  value={action.label}
                  color="fg.error"
                  disabled={action.disabled}
                  _hover={{ bg: 'bg.error', color: 'fg.error' }}
                >
                  <Icon as={action.icon} boxSize="3" />
                  <span>{action.label}</span>
                </Menu.Item>
              ))}
            </Menu.ItemGroup>
            <Menu.Separator />
            <Menu.ItemGroup>
              {queueProcessorActions.map((action, index) => (
                <Menu.Item key={index} disabled={action.disabled} onClick={action.onClick} value={action.label}>
                  <Icon as={action.icon} boxSize="3" />
                  <span>{action.label}</span>
                </Menu.Item>
              ))}
            </Menu.ItemGroup>
          </Menu.Content>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
