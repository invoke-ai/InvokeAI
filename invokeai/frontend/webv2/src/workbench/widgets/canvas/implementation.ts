import type { WidgetImplementation } from '@workbench/widgetContracts';

import { CanvasHeaderActions } from './CanvasHeaderActions';
import { CanvasWidgetView } from './CanvasWidgetView';

export const widgetImplementation = {
  headerActions: CanvasHeaderActions,
  view: CanvasWidgetView,
} satisfies WidgetImplementation;
