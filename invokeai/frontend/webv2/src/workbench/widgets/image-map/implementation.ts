import type { WidgetImplementation } from '@workbench/widgetContracts';

import { ImageMapHeaderActions } from './ImageMapHeaderActions';
import { ImageMapWidgetFooter } from './ImageMapWidgetFooter';
import { ImageMapWidgetView } from './ImageMapWidgetView';

export const widgetImplementation = {
  footer: ImageMapWidgetFooter,
  headerActions: ImageMapHeaderActions,
  view: ImageMapWidgetView,
} satisfies WidgetImplementation;
