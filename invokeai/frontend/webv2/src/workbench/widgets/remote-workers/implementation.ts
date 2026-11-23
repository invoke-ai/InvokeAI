import type { WidgetImplementation } from '@workbench/widgetContracts';

import { RemoteWorkersWidgetView } from './RemoteWorkersWidgetView';

export const widgetImplementation = { view: RemoteWorkersWidgetView } satisfies WidgetImplementation;
