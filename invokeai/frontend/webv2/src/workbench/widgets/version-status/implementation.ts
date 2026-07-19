import type { WidgetImplementation } from '@workbench/widgetContracts';

import { VersionStatusWidgetView } from './VersionStatusWidgetView';

export const widgetImplementation = { view: VersionStatusWidgetView } satisfies WidgetImplementation;
