import type { WorkbenchFocusController } from '@workbench/focusRegions';
import type { WidgetRegion } from '@workbench/layoutContracts';
import type { RegisteredWidget, WidgetTypeId } from '@workbench/widgetContracts';
import type { WorkbenchQueries, WorkbenchWidgetCommands } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { openWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import { revealPlacedWidget } from '@workbench/widgetPlacementCommands';

import type { NotificationActionContext } from './notificationActions';

export interface EditorNotificationActionSources {
  focus: Pick<WorkbenchFocusController, 'focusFloating' | 'focusRegion'>;
  getWidgetById: (widgetId: WidgetTypeId) => RegisteredWidget | undefined;
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  queries: Pick<WorkbenchQueries, 'getSnapshot'>;
  t: TFunction;
  widgets: WorkbenchWidgetCommands;
}

/** The context notification actions run in inside the editor: its store, widget registry and focus. */
export const createEditorNotificationActionContext = ({
  focus,
  getWidgetById,
  getWidgetsForRegion,
  queries,
  t,
  widgets,
}: EditorNotificationActionSources): NotificationActionContext => ({
  getWidgetLabel: (widgetId) => {
    const widget = getWidgetById(widgetId);

    return widget?.status === 'enabled' ? resolveWidgetLabel(widget.manifest, t) : null;
  },
  isProjectOpen: (projectId) => queries.getSnapshot().projects.some((project) => project.id === projectId),
  openWidget: (widgetId) => {
    // Where the layout already has the widget, bring that one forward rather than open another.
    const revealed = revealPlacedWidget({ project: queries.getSnapshot().activeProject, typeId: widgetId, widgets });

    if (revealed?.kind === 'floating') {
      focus.focusFloating(revealed.instanceId);
    } else if (revealed) {
      focus.focusRegion(revealed.region, widgetId);
    } else {
      const opened = openWorkbenchWidget(widgets, getWidgetsForRegion, widgetId);

      if (opened.ok) {
        focus.focusRegion(opened.region, widgetId);
      }
    }
  },
  t,
});
