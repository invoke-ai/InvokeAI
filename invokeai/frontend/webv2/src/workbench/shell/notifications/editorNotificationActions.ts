import type { WorkbenchFocusController } from '@workbench/focusRegions';
import type { WidgetRegion } from '@workbench/layoutContracts';
import type { RegisteredWidget, WidgetTypeId } from '@workbench/widgetContracts';
import type {
  WorkbenchNotificationCommands,
  WorkbenchQueries,
  WorkbenchWidgetCommands,
} from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { describeQueueRetry } from '@features/queue';
import { requestQueueItemReveal } from '@features/queue/reveal';
import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { revealOrOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { resolveWidgetLabel } from '@workbench/widgetLabels';

import type { NotificationActionContext } from './notificationActions';

export interface EditorNotificationActionSources {
  focus: Pick<WorkbenchFocusController, 'focusFloating' | 'focusRegion'>;
  getWidgetById: (widgetId: WidgetTypeId) => RegisteredWidget | undefined;
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  notifications: Pick<WorkbenchNotificationCommands, 'add' | 'withdrawRetry'>;
  queries: Pick<WorkbenchQueries, 'getSnapshot'>;
  /** Queue's retry command (`queueCommands.retryItems`): the ids of the items that re-run the given ones. */
  retryQueueItems: (itemIds: number[]) => Promise<number[]>;
  t: TFunction;
  widgets: WorkbenchWidgetCommands;
}

/**
 * Retries already asked for from a notice, by the set of backend items it retries: a rerun is queued once however many
 * notices, toasts or clicks offer it. Session-only, per account.
 */
const retryStates = new Map<string, 'done' | 'running'>();

const toRetryKey = (itemIds: readonly number[]): string => [...itemIds].sort((a, b) => a - b).join(',');

registerAccountOwnedResource({ clear: () => retryStates.clear(), name: 'notification-queue-retries' });

/** The context notification actions run in inside the editor: its store, widget registry and focus. */
export const createEditorNotificationActionContext = ({
  focus,
  getWidgetById,
  getWidgetsForRegion,
  notifications,
  queries,
  retryQueueItems,
  t,
  widgets,
}: EditorNotificationActionSources): NotificationActionContext => {
  const openWidget = (widgetId: WidgetTypeId): void =>
    revealOrOpenWorkbenchWidget({ focus, getWidgetsForRegion, queries, widgets }, widgetId);

  return {
    getWidgetLabel: (widgetId) => {
      const widget = getWidgetById(widgetId);

      return widget?.status === 'enabled' ? resolveWidgetLabel(widget.manifest, t) : null;
    },
    isProjectOpen: (projectId) => queries.getSnapshot().projects.some((project) => project.id === projectId),
    openWidget,
    revealQueueItem: (itemId) => {
      openWidget('queue');
      requestQueueItemReveal(itemId);
    },
    canRetryQueueItem: (itemIds) => !retryStates.has(toRetryKey(itemIds)),
    retryQueueItem: ({ canvasProjectId, itemIds }) => {
      const retryKey = toRetryKey(itemIds);
      // Once per set of backend items: every toast and list entry offering this retry shares the guard.
      if (retryStates.has(retryKey)) {
        return;
      }

      const owner = captureAccountScope();
      // English stands in for any key the catalog lacks; the record shows the active language.
      const english = (key: string, values?: Record<string, number>) => t(key, { ...values, lng: 'en' });

      retryStates.set(retryKey, 'running');
      retryQueueItems([...itemIds])
        .then((retriedItemIds) => {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          retryStates.set(retryKey, 'done');
          // The notices that offered it now say it is done; the report below links the first rerun.
          notifications.withdrawRetry(itemIds);
          const report = describeQueueRetry({
            isProjectClosedHere:
              canvasProjectId !== undefined &&
              !queries.getSnapshot().projects.some((project) => project.id === canvasProjectId),
            itemIds,
            retriedItemIds,
          });

          notifications.add({
            ...(report.rerunItemId === undefined
              ? {}
              : { actions: [{ itemId: report.rerunItemId, kind: 'reveal-queue-item' }] }),
            kind: report.kind,
            message: english(report.messageKey, report.values),
            messageKey: report.messageKey,
            messageValues: report.values,
            title: english(report.titleKey),
            titleKey: report.titleKey,
          });
        })
        .catch((error: unknown) => {
          // A newer account's guards were cleared with the old one; this answer belongs to neither.
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          // A failed attempt may be tried again.
          retryStates.delete(retryKey);
          notifications.add({
            kind: 'error',
            message: getApiErrorMessage(error, english('widgets.queue.couldNotRetryItem')),
            title: english('widgets.queue.retryFailed'),
            titleKey: 'widgets.queue.retryFailed',
          });
        });
    },
    t,
  };
};
