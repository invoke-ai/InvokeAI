import type { WidgetImplementation } from '@workbench/widgetContracts';

import { ShortcutGuide } from './ShortcutGuide';

export const widgetImplementation = { view: ShortcutGuide } satisfies WidgetImplementation;
