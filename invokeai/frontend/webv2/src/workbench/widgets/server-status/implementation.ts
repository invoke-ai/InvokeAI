import type { WidgetImplementation } from '@workbench/widgetContracts';

import { ServerStatusWidgetView } from './ServerStatusWidgetView';

export const widgetImplementation = { view: ServerStatusWidgetView } satisfies WidgetImplementation;
