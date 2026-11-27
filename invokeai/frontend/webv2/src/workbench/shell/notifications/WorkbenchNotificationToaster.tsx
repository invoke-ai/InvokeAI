import { queueCommands } from '@features/queue';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useWorkbenchFocus } from '@workbench/focusRegions';
import { getWorkbenchPreferences } from '@workbench/settings/store';
import { useWorkbenchCommands, useWorkbenchQueries, useWorkbenchSubscription } from '@workbench/WorkbenchContext';
import { useWorkbenchWidgetRegistry } from '@workbench/WorkbenchWidgetRegistryContext';
import { useTranslation } from 'react-i18next';

import { createEditorNotificationActionContext } from './editorNotificationActions';
import { startNotificationToasts } from './notificationToasts';

export const WorkbenchNotificationToaster = () => {
  const { i18n } = useTranslation();
  const queries = useWorkbenchQueries();
  const subscribe = useWorkbenchSubscription();
  const { notifications, widgets } = useWorkbenchCommands();
  const { getWidgetById, getWidgetsForRegion } = useWorkbenchWidgetRegistry();
  const focus = useWorkbenchFocus();

  // Everything read here lives as long as the editor: the store, the widget registry, the focus controller, the queue's
  // commands, and the i18n instance, whose `t` follows language changes.
  useMountEffect(() =>
    startNotificationToasts({
      actionContext: createEditorNotificationActionContext({
        focus,
        getWidgetById,
        getWidgetsForRegion,
        notifications,
        queries,
        retryQueueItems: queueCommands.retryItems,
        t: i18n.t,
        widgets,
      }),
      getNotifications: () => queries.getSnapshot().notifications,
      getPreferences: getWorkbenchPreferences,
      subscribe,
    })
  );

  return null;
};
