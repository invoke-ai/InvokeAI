import type { WidgetImplementation } from '@workbench/widgetContracts';

import { QueueStatusWidgetView } from './QueueStatusWidgetView';

export const widgetImplementation = { view: QueueStatusWidgetView } satisfies WidgetImplementation;
