import type { WidgetImplementation } from '@workbench/widgetContracts';

import { PreviewHeaderActions } from './PreviewHeaderActions';
import { PreviewWidgetLabel } from './PreviewWidgetChrome';
import { PreviewWidgetView } from './PreviewWidgetView';

export const widgetImplementation = {
  headerActions: PreviewHeaderActions,
  headerLabel: PreviewWidgetLabel,
  view: PreviewWidgetView,
} satisfies WidgetImplementation;
