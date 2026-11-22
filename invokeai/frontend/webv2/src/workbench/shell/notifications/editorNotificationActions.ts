import type { WorkbenchFocusController } from '@workbench/focusRegions';
import type { WidgetRegion } from '@workbench/layoutContracts';
import type { RegisteredWidget, WidgetTypeId } from '@workbench/widgetContracts';
import type { WorkbenchCommands, WorkbenchQueries } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { describeQueueRetry } from '@features/queue';
import { requestQueueItemReveal } from '@features/queue/reveal';
import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { getWorkbenchPreferences, patchWorkbenchPreferences } from '@workbench/settings/store';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import { addWidgetToWorkspace } from '@workbench/widgetPlacementCommands';
import { selectStructureLock } from '@workbench/workspace/structureLock';
import { customizeWorkspaceById, openInWorkspace, startCustomizing } from '@workbench/workspace/workspaceNavigation';

import type { NotificationActionContext } from './notificationActions';

export interface EditorNotificationActionSources {
  commands: Pick<WorkbenchCommands, 'layout' | 'notifications' | 'widgets'>;
  focus: Pick<WorkbenchFocusController, 'focusFloating' | 'focusRegion'>;
  getWidgetById: (widgetId: WidgetTypeId) => RegisteredWidget | undefined;
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  queries: Pick<WorkbenchQueries, 'getSnapshot' | 'getWorkspaceLayout'>;
  /** Queue's retry command (`queueCommands.retryItems`): the ids of the items that re-run the given ones. */
  retryQueueItems: (itemIds: number[]) => Promise<number[]>;
  t: TFunction;
}

/** Retries already asked for, keyed by the items retried, so a rerun queues once; session-only, per account. */
const retryStates = new Map<string, 'done' | 'running'>();

const toRetryKey = (itemIds: readonly number[]): string => [...itemIds].sort((a, b) => a - b).join(',');

registerAccountOwnedResource({ clear: () => retryStates.clear(), name: 'notification-queue-retries' });

/** The context notification actions run in inside the editor: its store, widget registry and focus. */
export const createEditorNotificationActionContext = ({
  commands,
  focus,
  getWidgetById,
  getWidgetsForRegion,
  queries,
  retryQueueItems,
  t,
}: EditorNotificationActionSources): NotificationActionContext => {
  const { notifications } = commands;
  const ports = { commands, focus, getWidgetsForRegion, queries };
  const openWidget = (widgetId: WidgetTypeId): void => void openInWorkspace(ports, { typeId: widgetId });
  const getLock = () => selectStructureLock(queries.getSnapshot(), getWorkbenchPreferences().rearrangeWithoutCustomize);

  return {
    addWidget: (widgetId, surface) => {
      const widget = getWidgetById(widgetId);

      if (!widget) {
        return;
      }
      // Locked, the add is the first edit of a Customize session; focus goes to the widget, not the bar.
      if (!getLock().canEditStructure) {
        startCustomizing({ commands, queries }, { focusBar: false });
      }
      // The add fronts it; the open that follows only moves focus there.
      if (addWidgetToWorkspace({ getWidgetsForRegion, surface, widget, widgets: commands.widgets })) {
        openInWorkspace(ports, { surface, typeId: widgetId }, { mode: 'stay' });
      }
    },
    customizeWorkspace: (workspaceId) => {
      if (!getLock().isCustomizing) {
        customizeWorkspaceById({ commands, queries }, workspaceId);
      }
    },
    isRearranging: () => getWorkbenchPreferences().rearrangeWithoutCustomize,
    lockWorkspaces: () => void patchWorkbenchPreferences({ rearrangeWithoutCustomize: false }),
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
