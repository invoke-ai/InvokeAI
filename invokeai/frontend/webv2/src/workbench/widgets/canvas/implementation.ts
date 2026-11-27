import type { WidgetImplementation } from '@workbench/widgetContracts';

import { CanvasHeaderActions, CanvasSettingsActions } from './CanvasHeaderActions';
import { CanvasWidgetView } from './CanvasWidgetView';

export const widgetImplementation = {
  headerActions: CanvasHeaderActions,
  settingsActions: CanvasSettingsActions,
  view: CanvasWidgetView,
} satisfies WidgetImplementation;
