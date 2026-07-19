import type { WidgetManifest } from '@workbench/widgetContracts';

import { CloudCheckIcon } from 'lucide-react';

export const autosaveStatusWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['bottom'],
  bottomPanel: 'tooltip',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: CloudCheckIcon,
  id: 'autosave-status',
  label: (t) => t('widgets.labels.autosaveStatus'),
  load: () => import('./implementation').then((module) => module.widgetImplementation),
  version: 1,
};
