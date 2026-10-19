import type { WidgetImplementation } from '@workbench/widgetContracts';

import { DiagnosticsHeaderActions } from './DiagnosticsHeaderActions';
import { DiagnosticsWidgetView } from './DiagnosticsWidgetView';

export const widgetImplementation = {
  headerActions: DiagnosticsHeaderActions,
  view: DiagnosticsWidgetView,
} satisfies WidgetImplementation;
