import type { WidgetImplementation } from '@workbench/widgetContracts';

import { LayersWidgetView } from './LayersWidgetView';

export const widgetImplementation = {
  view: LayersWidgetView,
} satisfies WidgetImplementation;
