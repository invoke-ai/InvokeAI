import type { WidgetImplementation } from '@workbench/widgetContracts';

import { AutosaveStatusWidgetView } from './AutosaveStatusWidgetView';

export const widgetImplementation = { view: AutosaveStatusWidgetView } satisfies WidgetImplementation;
