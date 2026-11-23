import type { QueueCounts } from '@features/queue/core/types';
import type { TFunction } from 'i18next';

import { Icon, Menu, Text } from '@chakra-ui/react';
import { getQueueItemReadModelLabel } from '@features/queue/core/itemLabels';
import { getPersonalQueueActivity } from '@features/queue/core/types';
import { queueCommands } from '@features/queue/publicApi';
import { noteQueueCancelRequested, type QueueCancelRequest } from '@features/queue/runtime/cancelRequests';
import { getApiErrorMessage } from '@platform/transport/http';
import { useQueryClient } from '@tanstack/react-query';
import {
  BrushCleaningIcon,
  ListOrderedIcon,
  PauseIcon,
  PlayIcon,
  Trash2Icon,
  TrashIcon,
  XIcon,
  type LucideIcon,
} from 'lucide-react';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { QueueUiNotificationPort, QueueViewer } from './QueueUiContext';

import { cancelCurrentQueueItem, getCancelCurrentTarget, getInProgressAnywhere } from './cancelCurrent';
import { requestQueueConfirmation, type QueueConfirmation } from './queueConfirmationStore';
import {
  refreshQueue,
  useIsProcessorPaused,
  useNowNextItems,
  useQueueCounts,
  useQueueReadModelData,
  useWholeQueueStatus,
} from './queueDataStore';
import {
  translateScoped,
  useQueueActionScope,
  useQueueQueryScope,
  useScopedLocalRuns,
  type QueueActionScope,
} from './queueScope';
import { useQueueUi } from './QueueUiContext';

export type QueueActionId =
  | 'cancel-current'
  | 'cancel-all'
  | 'cancel-all-except-current'
  | 'toggle-processor'
  | 'open-queue'
  | 'clear-failed'
  | 'clear-finished'
  | 'clear-queue';

export type QueueActionGroup = 'cancel' | 'processor' | 'clear';

export interface QueueAction {
  id: QueueActionId;
  group: QueueActionGroup;
  label: string;
  hint?: string;
  icon: LucideIcon;
  disabled: boolean;
  destructive: boolean;
  /** Asks first when the action is bulk or permanent; otherwise runs it. Every surface calls this, never `run`. */
  trigger: () => Promise<void>;
}

type QueueActionConfirmationCopy = Omit<QueueConfirmation, 'onConfirm'>;

interface QueueActionDescriptor extends Omit<QueueAction, 'trigger'> {
  /** Null runs at once. */
  confirmation: QueueActionConfirmationCopy | null;
}

/** Unfinished work is the viewer's own (everyone's for admins); history is null where the server cannot count it. */
export interface QueueActionCounts {
  history: { canceled: number; completed: number; failed: number; total: number } | null;
  hasHistory: boolean;
  inProgress: number;
  pending: number;
}

export const getQueueActionCounts = (
  counts: QueueCounts,
  scope: QueueActionScope,
  viewer: QueueViewer
): QueueActionCounts => {
  const unfinished =
    viewer === 'admin' ? { inProgress: counts.inProgress, pending: counts.pending } : getPersonalQueueActivity(counts);

  return {
    ...unfinished,
    hasHistory: counts.total > 0,
    // Without an origin prefix a member's status counts every user's items, which their clears do not reach.
    history:
      scope.kind === 'mine'
        ? null
        : { canceled: counts.canceled, completed: counts.completed, failed: counts.failed, total: counts.total },
  };
};

const getFinishedCount = (history: NonNullable<QueueActionCounts['history']>): number =>
  history.completed + history.canceled + history.failed;

const withRunningItemKept = (t: TFunction, text: string, keepsRunningItem: boolean): string =>
  keepsRunningItem ? `${text} ${t('widgets.queue.runningItemKeepsGoing')}` : text;

/**
 * Cancelling the running item never asks (Retry brings it back); bulk stops and deletes ask with scope and count.
 * Bulk cancels also stop this tab's unaccepted runs, so they stay available offline.
 */
export const getQueueActionDescriptors = ({
  canManageProcessor,
  counts,
  cancelCurrentTarget,
  hasRunningItem,
  includeOpenQueue,
  isConnected,
  isPaused,
  localUnsubmitted,
  scope,
  t,
}: {
  canManageProcessor: boolean;
  counts: QueueActionCounts;
  cancelCurrentTarget: 'in_progress' | 'pending' | null;
  hasRunningItem: boolean;
  includeOpenQueue: boolean;
  isConnected: boolean;
  isPaused: boolean;
  localUnsubmitted: number;
  scope: QueueActionScope;
  t: TFunction;
}): QueueActionDescriptor[] => {
  // Offline, the server's counts are a stale reading and its items out of reach; only local runs can stop.
  const unfinished = (isConnected ? counts.pending + counts.inProgress : 0) + localUnsubmitted;
  const queued = (isConnected ? counts.pending : 0) + localUnsubmitted;
  const { history } = counts;
  const finished = history ? getFinishedCount(history) : null;
  const keepRunning = t('widgets.queue.keepRunning');
  const keepItems = t('widgets.queue.keepItems');
  const cancelAllLabel = t('widgets.queue.cancelAllItems');
  const cancelAllExceptCurrentLabel = t('widgets.queue.cancelAllExceptCurrent');
  const clearFailedLabel = t('widgets.queue.clearFailedItems');
  const clearFinishedLabel = t('widgets.queue.clearFinishedItems');
  const clearQueueLabel = t('widgets.queue.clearAllItems');
  const clearQueueBody = translateScoped(t, 'widgets.queue.clearQueueConfirmationBody', scope, history?.total ?? null);
  const serverUnfinished = counts.pending + counts.inProgress;

  return [
    {
      confirmation: null,
      destructive: true,
      disabled: !isConnected || cancelCurrentTarget === null,
      group: 'cancel',
      icon: XIcon,
      id: 'cancel-current',
      label:
        cancelCurrentTarget === 'pending' ? t('widgets.queue.cancelNextItem') : t('widgets.queue.cancelCurrentItem'),
    },
    {
      confirmation: {
        body: translateScoped(t, 'widgets.queue.cancelAllConfirmationBody', scope, unfinished),
        cancelLabel: keepRunning,
        confirmLabel: cancelAllLabel,
        title: t('widgets.queue.actionConfirmationTitle', { action: cancelAllLabel }),
      },
      destructive: true,
      disabled: unfinished === 0,
      group: 'cancel',
      icon: XIcon,
      id: 'cancel-all',
      label: cancelAllLabel,
    },
    {
      confirmation: {
        body: withRunningItemKept(
          t,
          translateScoped(t, 'widgets.queue.cancelAllExceptCurrentConfirmationBody', scope, queued),
          isConnected && hasRunningItem
        ),
        cancelLabel: keepRunning,
        confirmLabel: cancelAllExceptCurrentLabel,
        title: t('widgets.queue.actionConfirmationTitle', { action: cancelAllExceptCurrentLabel }),
      },
      destructive: true,
      disabled: queued === 0,
      group: 'cancel',
      icon: XIcon,
      id: 'cancel-all-except-current',
      label: cancelAllExceptCurrentLabel,
    },
    {
      confirmation: null,
      destructive: false,
      disabled: !isConnected || !canManageProcessor,
      group: 'processor',
      ...(isPaused
        ? {
            hint: t('widgets.queue.processorPausedWaiting', { count: counts.pending }),
            icon: PlayIcon,
            label: t('widgets.queue.resumeProcessor'),
          }
        : { icon: PauseIcon, label: t('widgets.queue.pauseProcessor') }),
      id: 'toggle-processor',
    },
    ...(includeOpenQueue
      ? [
          {
            confirmation: null,
            destructive: false,
            disabled: false,
            group: 'processor' as const,
            icon: ListOrderedIcon,
            id: 'open-queue' as const,
            label: t('widgets.queue.openQueue'),
          },
        ]
      : []),
    {
      confirmation: {
        body: translateScoped(t, 'widgets.queue.clearFailedConfirmationBody', scope, history?.failed ?? null),
        cancelLabel: keepItems,
        confirmLabel: clearFailedLabel,
        title: t('widgets.queue.clearFailedTitle'),
      },
      destructive: true,
      disabled: !isConnected || (history ? history.failed === 0 : !counts.hasHistory),
      group: 'clear',
      icon: Trash2Icon,
      id: 'clear-failed',
      label: clearFailedLabel,
    },
    {
      confirmation: {
        body: translateScoped(t, 'widgets.queue.clearFinishedConfirmationBody', scope, finished),
        cancelLabel: keepItems,
        confirmLabel: clearFinishedLabel,
        title: t('widgets.queue.clearFinishedTitle'),
      },
      destructive: true,
      disabled: !isConnected || (finished === null ? !counts.hasHistory : finished === 0),
      group: 'clear',
      icon: BrushCleaningIcon,
      id: 'clear-finished',
      label: clearFinishedLabel,
    },
    {
      confirmation: {
        body:
          serverUnfinished > 0
            ? `${clearQueueBody} ${t('widgets.queue.clearQueueStopsWork', { count: serverUnfinished })}`
            : clearQueueBody,
        cancelLabel: keepItems,
        confirmLabel: clearQueueLabel,
        title: t('widgets.queue.clearQueueTitle'),
      },
      destructive: true,
      disabled: !isConnected || (history ? history.total === 0 : !counts.hasHistory),
      group: 'clear',
      icon: TrashIcon,
      id: 'clear-queue',
      label: clearQueueLabel,
    },
  ];
};

const reportFailure = (notify: QueueUiNotificationPort, title: string, fallback: string) => (error: unknown) =>
  notify.error(title, getApiErrorMessage(error, fallback));

/** Dropped when the request fails, so a later cancel by someone else still reports. */
const withCancelNote = async (request: QueueCancelRequest, send: () => Promise<void>): Promise<void> => {
  const drop = noteQueueCancelRequested(request);

  try {
    await send();
  } catch (error) {
    drop();
    throw error;
  }
};

export const useQueueActions = ({ includeOpenQueue = false }: { includeOpenQueue?: boolean } = {}): QueueAction[] => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const serverCounts = useQueueCounts();
  const { current } = useNowNextItems();
  const readModel = useQueueReadModelData();
  const isPaused = useIsProcessorPaused();
  const queryScope = useQueueQueryScope();
  const scope = useQueueActionScope();
  const {
    activeProjectId,
    canManageItem,
    canManageProcessor,
    cancelLocalRuns,
    isConnected,
    notify,
    openQueue,
    viewer,
  } = useQueueUi();
  const counts = useMemo(() => getQueueActionCounts(serverCounts, scope, viewer), [scope, serverCounts, viewer]);
  const hasRunningItem = current?.status === 'in_progress' && canManageItem(current);
  const wholeQueueStatus = useWholeQueueStatus(queryScope);
  const cancelCurrentTarget = useMemo(
    () =>
      getCancelCurrentTarget(readModel, {
        canManageItem,
        inProgressAnywhere: getInProgressAnywhere(queryScope, readModel, wholeQueueStatus),
      }),
    [canManageItem, queryScope, readModel, wholeQueueStatus]
  );
  const localProjectId = scope.kind === 'project' ? activeProjectId : null;
  const localUnsubmitted = useScopedLocalRuns().length;

  const runners = useMemo<Record<QueueActionId, () => Promise<void>>>(() => {
    // Each result repeats the counts its confirmation named: both come from the model the trigger was built with.
    const originPrefix = queryScope.originPrefix ?? null;
    /** Local runs are never the current item, so every bulk cancel stops all of them. */
    const cancelInBulk = async (
      serverCount: number,
      cancelOnServer: () => Promise<void>,
      describeRequested: (count: number) => string
    ): Promise<void> => {
      cancelLocalRuns(localProjectId);
      await (serverCount > 0 ? cancelOnServer().then(() => refreshQueue()) : Promise.resolve())
        .then(() =>
          notify.success(
            t('widgets.queue.queueCancellationRequested'),
            describeRequested(serverCount + localUnsubmitted)
          )
        )
        .catch(reportFailure(notify, t('widgets.queue.failedToCancelItems'), t('widgets.queue.couldNotCancelItems')));
    };

    return {
      'cancel-all': () =>
        cancelInBulk(
          isConnected ? counts.pending + counts.inProgress : 0,
          () => withCancelNote({ kind: 'scope', originPrefix }, () => queueCommands.cancelScopedItems(queryScope)),
          (count) => translateScoped(t, 'widgets.queue.cancelAllRequested', scope, count)
        ),
      'cancel-all-except-current': () =>
        cancelInBulk(
          isConnected ? counts.pending : 0,
          () =>
            withCancelNote({ exceptItemId: current?.id, kind: 'scope', originPrefix }, () =>
              queueCommands.cancelScopedItems(queryScope, { keepCurrent: true })
            ),
          (count) =>
            withRunningItemKept(
              t,
              translateScoped(t, 'widgets.queue.cancelAllExceptCurrentRequested', scope, count),
              isConnected && hasRunningItem
            )
        ),
      'cancel-current': async () => {
        const outcome = await cancelCurrentQueueItem({ canManageItem, client: queryClient, scope: queryScope });

        switch (outcome.kind) {
          case 'canceled':
            notify.success(
              t('widgets.queue.cancellationRequested'),
              getQueueItemReadModelLabel(outcome.item) || t('widgets.queue.itemTitle', { id: outcome.item.id })
            );
            return;
          case 'finished':
            notify.info(
              t('widgets.queue.itemAlreadyFinished'),
              getQueueItemReadModelLabel(outcome.item) || t('widgets.queue.itemTitle', { id: outcome.item.id })
            );
            return;
          case 'nothing':
            notify.info(
              t('widgets.queue.noCurrentItem'),
              translateScoped(t, 'widgets.queue.noCurrentItemDescription', scope)
            );
            return;
          case 'failed':
            reportFailure(
              notify,
              t('widgets.queue.failedToCancelCurrentItem'),
              t('widgets.queue.couldNotCancelCurrentItem')
            )(outcome.error);
            return;
          // A press while one is in flight, or one an account change overtook, reports nothing.
          case 'busy':
          case 'superseded':
            return;
        }
      },
      'clear-failed': async () => {
        await queueCommands
          .clearFailedItems(queryScope)
          .then(() => refreshQueue())
          .then(() =>
            notify.success(
              t('widgets.queue.failedItemsCleared'),
              translateScoped(t, 'widgets.queue.failedItemsClearedDescription', scope, counts.history?.failed ?? null)
            )
          )
          .catch(
            reportFailure(notify, t('widgets.queue.failedToClearItems'), t('widgets.queue.couldNotClearFailedItems'))
          );
      },
      'clear-finished': async () => {
        await queueCommands
          .clearFinishedItems(queryScope)
          .then(() =>
            notify.success(
              t('widgets.queue.finishedItemsCleared'),
              translateScoped(
                t,
                'widgets.queue.finishedItemsClearedDescription',
                scope,
                counts.history ? getFinishedCount(counts.history) : null
              )
            )
          )
          .catch(
            reportFailure(notify, t('widgets.queue.failedToClearItems'), t('widgets.queue.couldNotClearFinishedItems'))
          )
          // A scoped clear deletes item by item, so a failure part-way has still changed the list.
          .finally(() => refreshQueue());
      },
      'clear-queue': async () => {
        // Clearing deletes running work without a cancel event; its runs here must not report as failures.
        await withCancelNote({ kind: 'scope', originPrefix }, () => queueCommands.clearItems(queryScope))
          .then(() => refreshQueue())
          .then(() =>
            notify.success(
              t('widgets.queue.queueCleared'),
              translateScoped(t, 'widgets.queue.queueClearedDescription', scope, counts.history?.total ?? null)
            )
          )
          .catch(reportFailure(notify, t('widgets.queue.failedToClearQueue'), t('widgets.queue.couldNotClearQueue')));
      },
      'open-queue': () => {
        openQueue();
        return Promise.resolve();
      },
      'toggle-processor': async () => {
        const pause = !isPaused;

        await (pause ? queueCommands.pauseProcessor() : queueCommands.resumeProcessor())
          .then(() =>
            pause
              ? notify.success(
                  t('widgets.queue.processorPaused'),
                  t('widgets.queue.processorPausedDescription', { count: counts.pending })
                )
              : notify.success(t('widgets.queue.processorResumed'))
          )
          .then(() => refreshQueue())
          .catch(
            reportFailure(
              notify,
              pause ? t('widgets.queue.failedToPauseProcessor') : t('widgets.queue.failedToResumeProcessor'),
              t('widgets.queue.couldNotChangeProcessor')
            )
          );
      },
    };
  }, [
    canManageItem,
    cancelLocalRuns,
    counts,
    current,
    hasRunningItem,
    isConnected,
    isPaused,
    localProjectId,
    localUnsubmitted,
    notify,
    openQueue,
    queryClient,
    queryScope,
    scope,
    t,
  ]);

  return useMemo(
    () =>
      getQueueActionDescriptors({
        canManageProcessor,
        counts,
        cancelCurrentTarget:
          cancelCurrentTarget?.status === 'pending' ? 'pending' : cancelCurrentTarget ? 'in_progress' : null,
        hasRunningItem,
        includeOpenQueue,
        isConnected,
        isPaused,
        localUnsubmitted,
        scope,
        t,
      }).map(({ confirmation, ...action }) => {
        const run = runners[action.id];

        return {
          ...action,
          trigger: confirmation
            ? () => {
                requestQueueConfirmation({ ...confirmation, onConfirm: run });
                return Promise.resolve();
              }
            : run,
        };
      }),
    [
      cancelCurrentTarget,
      canManageProcessor,
      counts,
      hasRunningItem,
      includeOpenQueue,
      isConnected,
      isPaused,
      localUnsubmitted,
      runners,
      scope,
      t,
    ]
  );
};

export const getQueueAction = (actions: readonly QueueAction[], id: QueueActionId): QueueAction | undefined =>
  actions.find((action) => action.id === id);

const QueueActionMenuItem = ({ action }: { action: QueueAction }) => (
  <Menu.Item
    data-danger={action.destructive ? '' : undefined}
    disabled={action.disabled}
    value={action.id}
    onClick={action.trigger}
  >
    <Icon as={action.icon} boxSize="3" />
    <Menu.ItemText>
      {action.label}
      {action.hint ? (
        <Text as="span" color="fg.muted" display="block" fontSize="xs">
          {action.hint}
        </Text>
      ) : null}
    </Menu.ItemText>
  </Menu.Item>
);

const GroupLabel = ({ children }: { children: string }) => (
  <Menu.ItemGroupLabel color="fg.subtle" fontSize="xs" textTransform="uppercase">
    {children}
  </Menu.ItemGroupLabel>
);

export const QueueMenuItems = ({
  actions,
  includeClear = false,
  label,
}: {
  actions: QueueAction[];
  includeClear?: boolean;
  label?: string;
}) => {
  const { t } = useTranslation();
  const inGroup = (group: QueueActionGroup) => actions.filter((action) => action.group === group);

  return (
    <>
      <Menu.ItemGroup>
        {label ? <GroupLabel>{label}</GroupLabel> : null}
        {inGroup('cancel').map((action) => (
          <QueueActionMenuItem key={action.id} action={action} />
        ))}
      </Menu.ItemGroup>
      <Menu.Separator />
      <Menu.ItemGroup>
        {inGroup('processor').map((action) => (
          <QueueActionMenuItem key={action.id} action={action} />
        ))}
      </Menu.ItemGroup>
      {includeClear ? (
        <>
          <Menu.Separator />
          <Menu.ItemGroup>
            <GroupLabel>{t('common.clear')}</GroupLabel>
            {inGroup('clear').map((action) => (
              <QueueActionMenuItem key={action.id} action={action} />
            ))}
          </Menu.ItemGroup>
        </>
      ) : null}
    </>
  );
};
