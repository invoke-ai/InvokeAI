import type { WidgetImplementation } from '@workbench/widgetContracts';

import { ServerOutputHeaderActions, ServerOutputHeaderLabel } from './ServerOutputHeader';
import { ServerOutputWidgetView } from './ServerOutputWidgetView';

export const widgetImplementation = {
  headerActions: ServerOutputHeaderActions,
  headerLabel: ServerOutputHeaderLabel,
  view: ServerOutputWidgetView,
} satisfies WidgetImplementation;
