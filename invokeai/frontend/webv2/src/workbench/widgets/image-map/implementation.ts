import type { WidgetImplementation } from '@workbench/widgetContracts';

import { ImageMapDataRuntime } from './ImageMapDataRuntime';
import { ImageMapWidgetFooter } from './ImageMapWidgetFooter';
import { ImageMapWidgetView } from './ImageMapWidgetView';

export const widgetImplementation = {
  footer: ImageMapWidgetFooter,
  host: ImageMapDataRuntime,
  view: ImageMapWidgetView,
} satisfies WidgetImplementation;
