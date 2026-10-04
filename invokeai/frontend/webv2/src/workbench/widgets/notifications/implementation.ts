import type { WidgetImplementation } from '@workbench/widgetContracts';

import { NotificationsHeaderActions } from './NotificationsHeaderActions';
import { NotificationsWidgetView } from './NotificationsWidgetView';

export const widgetImplementation = {
  headerActions: NotificationsHeaderActions,
  view: NotificationsWidgetView,
} satisfies WidgetImplementation;
