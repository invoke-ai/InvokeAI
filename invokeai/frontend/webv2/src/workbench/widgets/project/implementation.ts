import type { WidgetImplementation } from '@workbench/widgetContracts';

import { ProjectWidgetView } from './ProjectWidgetView';

export const widgetImplementation = { view: ProjectWidgetView } satisfies WidgetImplementation;
