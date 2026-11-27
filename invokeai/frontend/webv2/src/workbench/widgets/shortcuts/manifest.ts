import type { WidgetManifest } from '@workbench/widgetContracts';

import { KeyboardIcon } from 'lucide-react';

export const shortcutsWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['bottom'],
  bottomPanel: 'popover',
  preserveWorkbenchFocus: true,
  compactSizing: 'remaining-space',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: KeyboardIcon,
  id: 'shortcuts',
  label: (t) => t('workbench.shortcuts.title'),
  load: () => import('./implementation').then((module) => module.widgetImplementation),
  version: 1,
};
