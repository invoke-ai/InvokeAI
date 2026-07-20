import type { WidgetImplementation } from '@workbench/widgetContracts';

import { LayersHeaderActions } from './LayersHeaderActions';
import { LayersWidgetView } from './LayersWidgetView';

export const widgetImplementation = {
  headerActions: LayersHeaderActions,
  view: LayersWidgetView,
} satisfies WidgetImplementation;
