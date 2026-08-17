import type { WidgetImplementation } from '@workbench/widgetContracts';

import { ImageMapDataRuntime } from './ImageMapDataRuntime';
import { ImageMapHeaderActions } from './ImageMapHeaderActions';
import { ImageMapWidgetFooter } from './ImageMapWidgetFooter';
import { ImageMapWidgetView } from './ImageMapWidgetView';

export const widgetImplementation = {
  footer: ImageMapWidgetFooter,
  headerActions: ImageMapHeaderActions,
  host: ImageMapDataRuntime,
  view: ImageMapWidgetView,
} satisfies WidgetImplementation;
