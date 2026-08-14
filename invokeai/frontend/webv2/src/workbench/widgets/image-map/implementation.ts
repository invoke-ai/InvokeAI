import type { WidgetImplementation } from '@workbench/widgetContracts';

import { ImageMapWidgetView } from './ImageMapWidgetView';

export const widgetImplementation = {
  view: ImageMapWidgetView,
} satisfies WidgetImplementation;
